"""Dyadic scale profiles, exact mask moments, and conservative MC ranks.

NumPy arrays end in (N,N), N a power of two. Float64 throughout.
No patient data, fitted masks, estimated noise covariance, or learned parameters.
"""
import numpy as np


def dimensions(N):
    m = int(np.log2(N))
    if N < 2 or 2**m != N:
        raise ValueError("N must be a power of two >=2")
    return np.array([1] + [3 * 4**(k-1) for k in range(1, m+1)])


def energies(x):
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[-1]
    if x.shape[-2] != N:
        raise ValueError("square arrays required")
    dimensions(N)
    a = x
    fine = []
    while a.shape[-1] > 1:
        x00, x01 = a[..., ::2, ::2], a[..., ::2, 1::2]
        x10, x11 = a[..., 1::2, ::2], a[..., 1::2, 1::2]
        h = (x00+x01-x10-x11)/2
        v = (x00-x01+x10-x11)/2
        t = (x00-x01-x10+x11)/2
        fine.append(np.sum(h*h+v*v+t*t, axis=(-2, -1)))
        a = (x00+x01+x10+x11)/2
    return np.stack([a[..., 0, 0]**2] + fine[::-1], axis=-1)


def profiles(x):
    e = energies(x)
    den = e.sum(axis=-1, keepdims=True)
    if np.any(den <= 0):
        raise ValueError("profile undefined at zero")
    return e / den


def projection(x, r):
    x = np.asarray(x, dtype=float)
    N = x.shape[-1]
    g = 2**r
    s = N//g
    a = x.reshape(x.shape[:-2]+(g, s, g, s)).mean(axis=(-3, -1))
    return np.repeat(np.repeat(a, s, axis=-2), s, axis=-1)


def details(x):
    p = [projection(x, r) for r in range(int(np.log2(x.shape[-1]))+1)]
    return np.stack([p[0]]+[p[k]-p[k-1] for k in range(1, len(p))])


def preprocess(x, mask, centered=False, kind="identity"):
    a = np.asarray(x, dtype=float)
    N = a.shape[-1]
    if kind == "hetero":
        scales = np.linspace(.5, 2., N)[None, :]
        a = a*scales
    elif kind == "smooth":
        # Fixed separable periodic linear smoothing; included in B for every ref.
        a = (2*a+np.roll(a, 1, -1)+np.roll(a, -1, -1))/4
        a = (2*a+np.roll(a, 1, -2)+np.roll(a, -1, -2))/4
    elif kind == "interpolate":
        # Periodic bilinear resampling by half a pixel along each axis.
        a = (a+np.roll(a, 1, -1))/2
        a = (a+np.roll(a, 1, -2))/2
    elif kind != "identity":
        raise ValueError(kind)
    a = a*mask
    if centered:
        a = (a-a.sum(axis=(-2, -1), keepdims=True)/mask.sum())*mask
    return a


def mask_gram(mask):
    """H_kj=tr(D_k T D_j T), O(n+m^2), no dense spatial matrices."""
    N = mask.shape[-1]
    n = N*N
    d = dimensions(N)
    L = mask.sum()
    e = energies(mask)
    h = np.zeros((len(d), len(d)))
    for j in range(1, len(d)):
        h[:j, j] = d[:j]/n*e[j]
        h[j, :j] = h[:j, j]
    np.fill_diagonal(h, L*d/n-h.sum(axis=1))
    return h


def exact_moments_direct(mask, centered=False):
    N = mask.shape[-1]
    n = N*N
    L = int(mask.sum())
    ell = L-int(centered)
    if ell < 1:
        raise ValueError("preprocessing has rank zero")
    d = dimensions(N)
    h = mask_gram(mask)
    a = L*d/n
    if centered:
        dm = details(mask).reshape(len(d), -1)
        e = energies(mask)
        g = (dm*mask.ravel())@dm.T
        h = h-2*g/L+np.outer(e, e)/(L*L)
        a = a-e/L
    mean = a/ell
    cov = 2/(ell*(ell+2))*(h-np.outer(a, a)/ell)
    # Enforce theoretical zeros at floating-point precision, not sample fit.
    if centered:
        mean[0] = 0
        cov[0, :] = cov[:, 0] = 0
    return mean, (cov+cov.T)/2


def _sum_children(a):
    return a[0::2, 0::2]+a[0::2, 1::2]+a[1::2, 0::2]+a[1::2, 1::2]


def mask_detail_gram_tree(mask):
    """g_kj=sum M (D_k M)(D_j M), O(n+m^2), using node means.

    No full-resolution array per level is constructed. Counting measure is used.
    """
    mask = np.asarray(mask, dtype=np.float64)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("a single square mask is required")
    dimensions(mask.shape[0])
    levels = [mask]
    while levels[-1].shape[0] > 1:
        levels.append(_sum_children(levels[-1])/4)
    levels.reverse()
    m = len(levels)-1
    n = mask.size
    gram = np.zeros((m+1,m+1), dtype=np.float64)
    L = float(mask.sum())
    gram[0,0] = L*float(levels[0][0,0])**2
    for j in range(1,m+1):
        parent = np.repeat(np.repeat(levels[j-1],2,axis=0),2,axis=1)
        dj = levels[j]-parent
        area = n/(4**j)
        gram[j,j] = area*np.sum(levels[j]*dj*dj)
        accum = area*_sum_children(dj*dj)
        for k in range(j-1,-1,-1):
            if k == 0:
                dk = levels[0]
            else:
                dk = levels[k]-np.repeat(np.repeat(levels[k-1],2,axis=0),2,axis=1)
            gram[k,j] = gram[j,k] = np.sum(dk*accum)
            if k:
                accum = _sum_children(accum)
    return gram


def exact_moments(mask, centered=False):
    """Binary-mask normalized moments using the linear-time tree algorithm."""
    mask = np.asarray(mask, dtype=np.float64)
    N = mask.shape[-1]
    d = dimensions(N)
    L = float(mask.sum())
    ell = L-int(centered)
    if ell < 1:
        raise ValueError("preprocessing has rank zero")
    h = mask_gram(mask)
    a = L*d/(N*N)
    if centered:
        e = energies(mask)
        g = mask_detail_gram_tree(mask)
        h = h-2*g/L+np.outer(e,e)/(L*L)
        a = a-e/L
    mean = a/ell
    cov = 2/(ell*(ell+2))*(h-np.outer(a,a)/ell)
    if centered:
        mean[0] = 0
        cov[0,:] = cov[:,0] = 0
    return mean, (cov+cov.T)/2


def iid_energy_covariance(mask, kappa, sigma=1., centered=False):
    """Unnormalized covariance for iid unit-variance input with excess kurtosis.

    This function does not supply non-Gaussian normalized-profile moments.
    """
    mask = np.asarray(mask, dtype=np.float64)
    N = mask.shape[-1];n=N*N;d=dimensions(N);L=float(mask.sum())
    if L < 1 or (centered and L < 2):
        raise ValueError("preprocessing has rank zero")
    h = mask_gram(mask)
    if centered:
        e=energies(mask);g=mask_detail_gram_tree(mask)
        h=h-2*g/L+np.outer(e,e)/L**2
        t=d/n+e/L**2
        diagonal_gram=(L*np.outer(t,t)-2*(np.outer(t,e)+np.outer(e,t))/L+4*g/L**2)
    else:
        diagonal_gram=L*np.outer(d,d)/n**2
    return sigma**4*(2*h+float(kappa)*diagonal_gram)


def statistics(y, mask, centered=False, with_q=True, profile_reference=None):
    p = profiles(y)
    u = dimensions(y.shape[-1])/(y.shape[-1]**2) if profile_reference is None else np.asarray(profile_reference, dtype=float)
    if u.shape != (p.shape[-1],) or np.any(u < 0) or not np.isclose(u.sum(), 1):
        raise ValueError("profile_reference must be a probability vector of matching dimension")
    h = .5*np.sum((np.sqrt(p)-np.sqrt(u))**2, axis=-1)
    peak = mask.sum()*np.max(y*y, axis=(-2, -1))/np.sum(y*y, axis=(-2, -1))
    if with_q:
        mu, cov = exact_moments(mask, centered)
        q = np.einsum("...i,ij,...j->...", p-mu, np.linalg.pinv(cov, rcond=1e-11), p-mu)
    else:
        q = np.zeros_like(h)
    return np.stack([h, q, peak], axis=-1)


METHODS = ["H_upper", "H_two", "Q_upper", "peak", "combined"]


def pvalues(observed, references):
    """Last coordinate: H, Q, peak. Common independent reference bank.

    Enlarged numerical tie sets can only increase the corresponding exact-tail
    p-values, modulo the usual accuracy of the computed statistics themselves.
    """
    refs = np.asarray(references)
    obs = np.atleast_2d(observed)
    out = np.empty((len(obs), 5))
    b = len(refs)
    for i, t in enumerate(obs):
        tol = 1e-12*np.maximum(1., np.abs(t))
        up = (1+(refs >= t-tol).sum(axis=0))/(b+1)
        lo = (1+(refs <= t+tol).sum(axis=0))/(b+1)
        out[i] = [up[0], min(1., 2*min(up[0], lo[0])), up[1], up[2],
                  min(1., 3*min(up[0], lo[0], up[2]))]
    return out


def make_masks(N):
    t = (np.arange(N)+.5)/N*2-1
    x, y = np.meshgrid(t, t)
    rng = np.random.default_rng(811+N)
    return {
        "full": np.ones((N,N)),
        "circle": ((x*x+y*y)<.65**2).astype(float),
        "irregular": ((x+.28*np.sin(4*y))<.1).astype(float),
        "half": (x<0).astype(float),
        "checker": ((np.indices((N,N)).sum(axis=0)%2)==0).astype(float),
        "random": (rng.random((N,N))<.5).astype(float),
        "one_pixel": np.eye(1, N, N//2).repeat(N, axis=0)*
                     (np.arange(N)==N//2)[:, None],
    }


def phantom(N):
    """Explicit five-ellipse mathematical phantom, not a patient image."""
    t = (np.arange(N)+.5)/N*2-1
    x, y = np.meshgrid(t, t)
    out = np.zeros((N,N))
    ellipses = [(1., 0., 0., .72, .9, 0.),
                (-.7, 0., -.02, .66, .82, 0.),
                (.35, -.25, .05, .18, .4, -.3),
                (.5, .28, .1, .15, .29, .35),
                (.25, 0., -.4, .2, .1, 0.)]
    for val, cx, cy, a, b, theta in ellipses:
        xx = (x-cx)*np.cos(theta)+(y-cy)*np.sin(theta)
        yy = -(x-cx)*np.sin(theta)+(y-cy)*np.cos(theta)
        out += val*((xx/a)**2+(yy/b)**2 <= 1.)
    return out


def signals(N):
    t = (np.arange(N)+.5)/N*2-1
    x,y = np.meshgrid(t,t)
    spike = np.zeros((N,N)); spike[N//2, N//2] = 1
    return {"step": (x>0).astype(float), "checker": (-1.)**np.indices((N,N)).sum(axis=0),
            "bump": np.exp(-(x*x+y*y)/.12), "spike": spike, "phantom": phantom(N)}


def prepare_q_metric(mean, covariance, rcond=1e-11):
    """Freeze a symmetric, scaled spectral metric before observation/reference use.

    Returns a callable and a serializable diagnostic record. Tiny negative
    modes are clipped only within a documented numerical tolerance. This is
    a numerical policy, not a forward-error theorem for exact_moments.
    Historical experiments retain their SVD/einsum statistic unchanged.
    """
    mean = np.asarray(mean, dtype=np.float64)
    c = np.asarray(covariance, dtype=np.float64)
    if c.shape != (mean.size, mean.size) or not np.all(np.isfinite(c)):
        raise ValueError('finite covariance of matching dimension required')
    if not 0 < rcond < 1 or not np.all(np.isfinite(mean)):
        raise ValueError('finite mean and 0 < rcond < 1 required')
    c = (c+c.T)/2
    scale = float(np.linalg.norm(c, 2))
    tolerance = 64*mean.size*np.finfo(np.float64).eps
    if scale == 0:
        eigenvalues = np.zeros(mean.size)
        weights = np.zeros((mean.size,0))
    else:
        eigenvalues, vectors = np.linalg.eigh(c/scale)
        if eigenvalues[0] < -tolerance:
            raise ValueError('covariance has a materially negative eigenvalue')
        clipped = np.maximum(eigenvalues,0.)
        active = clipped > rcond*clipped[-1]
        weights = vectors[:,active]/np.sqrt(scale*clipped[active])[None,:]
    frozen_mean = mean.copy()
    def evaluate(p):
        p = np.asarray(p,dtype=np.float64)
        if p.shape[-1] != frozen_mean.size or not np.all(np.isfinite(p)):
            raise ValueError('finite matching profiles required')
        return np.sum(((p-frozen_mean)@weights)**2,axis=-1)
    record = dict(scale=scale, scaled_eigenvalues=eigenvalues.tolist(),
                  relative_cutoff=float(rcond), retained_rank=weights.shape[1],
                  negative_tolerance=float(tolerance), contraction='sum of squared factor coordinates')
    return evaluate, record
