# Environment release notes (for maintainers)

The public `environment.yml` and `requirements.txt` were curated from the Conda and pip snapshots exported on **2026-11-05** from the author's working `openmmlab` environment. They have **not been validated by rebuilding the environment from scratch**.

## Source facts

- Python 3.8.20, PyTorch 2.1.2, CUDA runtime 11.8, GPU RTX 4060 Laptop.
- MMCV 2.1.0; MMEngine 0.10.7; MMDetection 3.3.0; MMDetection3D 1.4.0.
- SpConv-cu118 2.3.6; cumm-cu118 0.4.11; torch-scatter 2.1.2+pt21cu118.
- `pip freeze` contained an invalid machine-dependent editable MMDetection3D entry (`-e d:\\anaconda3\\envs\\openmmlab\\lib\\site-packages`); it is omitted from release files and must **not** be copied into `requirements.txt`.
- Original Conda export included duplicate Tsinghua channels, the retired `/pkgs/free` path and `cuda-version=12.9` mixed with CUDA 11.8 runtime components. These channels and CUDA toolkit runtime transitive pins are not reproduced in the curated environment file.

## Important limitations

- Minimal Conda specs and curated pip packages are *installation guidance*, not a tested portable lockfile.
- The author must verify the exact provenance/commit of their editable `mmdet3d` installation (the upstream v1.4.0 tag may differ from local files).
- Compiled extension installation on Windows can require MSVC and CUDA toolkit/build tooling; `mmcv` native ops and `torch-scatter` are version-sensitive.
- Before paper publication: verify creation in a new environment, smoke-test model import, then reproduce evaluation on the fixed test set and checkpoint.
- Do not accidentally commit the unfiltered snapshots containing private machine paths; keep them as internal audit records unless fully sanitized and explicitly labeled as snapshots.
