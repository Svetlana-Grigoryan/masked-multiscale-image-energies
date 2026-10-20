# Upload and archive this code-only package

1. Have all relevant authors approve code publication and settle a software license. Check third-party data notices.
2. On GitHub select **New repository**, name e.g. `masked-multiscale-image-energies`, and create an empty repository (avoid extra generated README/license if uploading local package).
3. Extract this ZIP and upload **its contents** to the repository, not the containing parent directory. For a larger file set, using Git is more reliable than browser upload.
4. Recommended CLI from the extracted directory: `git init && git add . && git commit -m "Initial code and data release" && git branch -M main && git remote add origin https://github.com/YOUR-ACCOUNT/masked-multiscale-image-energies.git && git push -u origin main` (GitHub repository must already exist).
5. Run the smoke test and inspect the public file list before making a release.
6. On GitHub select **Releases > Draft a new release**, create tag `v1.0.0`, and publish it.
7. For GitHub-integrated archival, enable Zenodo's GitHub integration for the repository BEFORE publishing a new release; if the first release was already made, publish a fresh release after enabling integration. Alternatively upload this ZIP directly to Zenodo as a software record. Avoid creating two independent records for the same version.
8. In Zenodo, verify creator order, metadata, license, related repository identifier, file contents, and publish. Zenodo then assigns actual DOIs. Update `CITATION.cff` and README using the issued DOI in a later version; avoid rewriting already deposited archives.

The source ZIP contains no manuscript files. Note that saved numerical metadata could preserve historical references to paper-level experiments.
