# Landsat mask-covariance computational pilot

A separate, executed pilot supplement to the Walsh–Haar manuscript; original manuscript results are not overwritten. See PROTOCOL_RU.md for the complete fixed design.

Install requirements.txt, then run from the extracted package:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/run_pilot.py
    python code/verify_pilot.py
    python code/make_report.py

Individual stages are --stage prepare, benchmark, moments, audit, power. Running all stages overwrites this pilot's outputs, not the prior manuscript records. The source TIFF is included under CC0; offline runs do not download it.

The data are a USGS Landsat 7 ETM-derived RGB fixture, not calibrated raw sensor observations. The 1024 square contains padding; 2048 benchmarking repeats a mask and does not create a higher-resolution real scene. All statistical claims concern explicitly simulated additional noise conditional on the frozen image and masks. No recovery or original-instrument-noise guarantee is asserted.
