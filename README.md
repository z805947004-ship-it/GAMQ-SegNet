# GAMQ-SegNet: Semantic and Instance Segmentation of 3D Maize Plants

GAMQ-SegNet is a 3D point-cloud segmentation framework for maize plant structural analysis. It extends [OneFormer3D](https://github.com/filaPro/oneformer3d) with a **GAASE** module in the sparse-convolutional backbone and **AMSQF**-related modifications in the query decoder, targeting semantic and instance segmentation of maize stems and leaves.

> **Reproducibility status.** The environment versions below were taken from the author's running training environment on **5 November 2026**. The *curated, fresh-install instructions* have **not yet been independently validated** on a clean machine. Dataset, trained checkpoint and final publication links are pending. Do not treat the repository as a complete paper reproduction package until those are supplied and the installation and evaluation steps have been tested.

## Features

- XYZRGB point clouds for maize-plant segmentation.
- Stem/leaf semantic segmentation and plant-part instance segmentation.
- **GAASE** backbone implementation: `oneformer3d/spconv_unet.py`.
- Query-decoder changes associated with **AMSQF**: `oneformer3d/query_decoder.py`.
- Training, evaluation and prediction-export scripts derived from the OneFormer3D framework.

## Repository structure

```text
GAMQ-SegNet/
├── configs/
│   └── oneformer3d_corn_formal_rgb_full.py
├── oneformer3d/
│   ├── spconv_unet.py
│   ├── query_decoder.py
│   ├── oneformer3d.py
│   └── ...
├── tools/
│   ├── train.py
│   ├── test.py
│   ├── infer_save_ply_corn.py
│   └── ...
├── environment.yml
├── requirements.txt
├── README.md
└── LICENSE
```

The `oneformer3d` package name is kept to preserve the upstream import paths.

## Experimental environment

The following versions were **observed in the working Windows environment**; they are *not* the versions in OneFormer3D's original Docker image.

| Component | Recorded version |
|---|---|
| OS | Windows (author's environment) |
| Python | **3.8.20** |
| PyTorch | **2.1.2** |
| PyTorch CUDA runtime | **11.8** |
| torchvision / torchaudio | **0.16.2 / 2.1.2** |
| MMCV | **2.1.0** |
| MMEngine | **0.10.7** |
| MMDetection | **3.3.0** |
| MMDetection3D | **1.4.0** (locally editable installation) |
| SpConv / cumm | **spconv-cu118 2.3.6 / cumm-cu118 0.4.11** |
| torch-scatter | **2.1.2+pt21cu118** |
| NumPy | **1.24.3** |
| Open3D | **0.19.0** |
| GPU used by the author | **NVIDIA GeForce RTX 4060 Laptop GPU** |

**Important:** OneFormer3D's published Dockerfile uses an older environment and MMDetection3D 1.1.0-era code. This project records the *author's actual installed MMDetection3D 1.4.0* instead. The original `pip freeze` also contained an editable entry pointing into the author's local `site-packages`; that machine-specific reference is deliberately **not** included here. The exact MMDetection3D source revision and any local modifications still need to be documented. CUDA extensions are especially sensitive to platform and compiler versions.

## Installation (Windows / NVIDIA CUDA 11.8)

Commands below use **PowerShell**. Linux installation may require different build steps.

### 1. Clone and create the base Conda environment

```powershell
git clone https://github.com/z805947004-ship-it/GAMQ-SegNet.git
cd GAMQ-SegNet
conda env create -f environment.yml
conda activate gamq-segnet
python -m pip install -r requirements.txt
```

If Conda dependency solving fails, resolve the package-channel/build availability instead of silently switching PyTorch/CUDA versions. Keep PyTorch **2.1.2** and CUDA **11.8** for parity with the reported environment.

### 2. Install and verify MMCV CUDA operations

```powershell
mim install "mmcv==2.1.0"
```

`mmcv` includes native operators. Depending on the current host platform, prebuilt binaries for this precise combination may not be available; **compilation with compatible CUDA/MSVC may be necessary**. Verify the installation before proceeding:

```powershell
python -c "import mmcv; from mmcv.ops import nms; print('MMCV', mmcv.__version__, 'native ops OK')"
```

### 3. Install sparse convolution and scatter extensions

```powershell
python -m pip install "spconv-cu118==2.3.6" "cumm-cu118==0.4.11"
python -m pip install --no-index --find-links https://data.pyg.org/whl/torch-2.1.0+cu118.html "torch-scatter==2.1.2+pt21cu118"
```

The PyG wheel index is for the **PyTorch 2.1 / CUDA 11.8** binary family; the URL uses `torch-2.1.0` even though the recorded patch release is PyTorch 2.1.2. Wheel availability varies by OS/Python architecture.

### 4. Install MMDetection3D 1.4.0

The working environment reported an **editable** MMDetection3D 1.4.0 install, but the exported editable path was local to the author's machine. To attempt a clean public installation, use the upstream source:

```powershell
cd ..
git clone --branch v1.4.0 https://github.com/open-mmlab/mmdetection3d.git mmdetection3d-v1.4.0
cd mmdetection3d-v1.4.0
python -m pip install -v --no-deps -e .
cd ..\GAMQ-SegNet
```

This **upstream v1.4.0 tag is a proposed clean-install reference**, not proof that the author's original editable installation was identical to that tag. If the code relies on locally modified MMDetection3D operators or APIs, document and publish the patch. Native extensions may need MSVC and a matching CUDA toolkit on Windows.

### 5. Verify core dependencies

```powershell
python -c "import torch, mmcv, mmdet, mmengine, mmdet3d, spconv, torch_scatter; from mmcv.ops import nms; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('mmcv', mmcv.__version__, 'mmdet', mmdet.__version__, 'mmdet3d', mmdet3d.__version__, 'mmengine', mmengine.__version__); print('CUDA extensions imported successfully')"
```

Successful imports do not by themselves prove full training/evaluation reproducibility. Run one actual training or inference smoke test using the dataset and checkpoint after installation.

## Dataset

The maize plant point cloud dataset used in this study is provided in the `data/` directory of this repository.

The dataset is intended for the training and evaluation of GAMQ-SegNet for semantic and instance segmentation of maize plants.

**Data Availability:** The dataset is included directly in this GitHub repository. No additional download from external cloud storage services is required.

Please refer to the dataset configuration files in `configs/` for the corresponding data loading settings.

## Training

```powershell
python tools/train.py configs/oneformer3d_corn_formal_rgb_full.py --work-dir work_dirs/gamq_segnet
```

The command follows the repository's reported training entry point; it must be verified with the released data paths and environment. Check the configuration for the actual training schedule, seed and model settings used in the article. Record the final checkpoint and relevant training log.

## Evaluation

```powershell
python tools/test.py configs/oneformer3d_corn_formal_rgb_full.py "PATH_TO_CHECKPOINT.pth" --work-dir work_dirs/gamq_segnet_test
```

Use the **same trained checkpoint and data split as the paper**. Exported numerical results and their evaluation protocol should be published alongside the dataset or in the repository.

A per-point prediction export utility is provided at `tools/infer_save_ply_corn.py`; consult its real command-line options before running it.

## Pretrained models and results

- GAMQ-SegNet checkpoint: **TODO (URL, SHA256)**
- Baselines and ablation models: **TODO (configurations and availability)**
- Evaluation outputs and summary metrics: **TODO**

Large datasets and model weights should be hosted separately (for example, on Zenodo) rather than tracked in ordinary Git commits.

## Relationship to upstream software and license

This work is derived from the [OneFormer3D repository](https://github.com/filaPro/oneformer3d) and its associated paper:

> Kolodiazhnyi, M., Vorontsova, A., Konushin, A., and Rukhovich, D. **OneFormer3D: One Transformer for Unified Point Cloud Segmentation.** CVPR 2024. [Paper](https://arxiv.org/abs/2311.14405)

Keep the required copyright notices and review the upstream license before licensing or redistributing modified upstream code. The presence of a `LICENSE` file in this repository does not establish that all third-party components share the same license.

## Citation

**TODO:** Insert the final GAMQ-SegNet paper's authors, title, venue, publication year, DOI and BibTeX when available. Please cite the original OneFormer3D publication when using the derived code.

## Contact

For reproducibility questions, please open a GitHub Issue. **TODO:** Add an approved author contact email if desired.
