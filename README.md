# GAMQ-SegNet: 3D Maize Plant Semantic and Instance Segmentation

**Related research:** *Three-Dimensional Reconstruction and Organ Segmentation of Maize Seedlings Using a Three-Axis RGB-D Acquisition Platform and GAMQ-SegNet*

GAMQ-SegNet is a 3D point-cloud segmentation method for maize stems and leaves, built on [OneFormer3D](https://github.com/filaPro/oneformer3d). It introduces **GAASE** (geometry-guided adaptive anisotropic sparse feature extraction) in the sparse-convolution backbone and **AMSQF** (adaptive multi-scale query fusion) in the query decoder.

This repository contains the **segmentation code, processed maize point-cloud data, and a trained checkpoint for evaluation**. The RGB-D acquisition platform and 2DGS reconstruction are part of the broader research workflow; raw RGB-D sequences are not included here.

## Repository layout

```text
GAMQ-SegNet/
├── configs/
│   └── oneformer3d_corn_formal_rgb_full.py
├── oneformer3d/
│   ├── spconv_unet.py          # GAASE
│   ├── query_decoder.py       # AMSQF
│   └── ...
├── tools/
│   ├── train.py
│   ├── test.py
│   └── infer_save_ply_corn.py
├── data/                       # Maize point-cloud dataset
│   ├── 1149-1(2)_pts.npy
│   ├── 1149-1(2)_sem.npy
│   ├── 1149-1(2)_ins.npy
│   └── ...
├── work_dirs/
│   └── oneformer3d-full-new/
│       └── best_all_ap_50%_epoch_660.pth
├── environment.yml
├── requirements.txt
├── LICENSE
└── README.md
```

## Environment

The experiments were conducted on **Windows** with an **NVIDIA GeForce RTX 4060 Laptop GPU**.

| Package | Version |
|---|---|
| Python | 3.8.20 |
| PyTorch / CUDA runtime | 2.1.2 / 11.8 |
| MMCV / MMEngine | 2.1.0 / 0.10.7 |
| MMDetection / MMDetection3D | 3.3.0 / 1.4.0 |
| spconv-cu118 / cumm-cu118 | 2.3.6 / 0.4.11 |
| torch-scatter | 2.1.2+pt21cu118 |
| Open3D | 0.19.0 |

### Installation (Windows PowerShell)

```powershell
git clone https://github.com/z805947004-ship-it/GAMQ-SegNet.git
cd GAMQ-SegNet
conda env create -f environment.yml
conda activate gamq-segnet
python -m pip install -r requirements.txt
mim install "mmcv==2.1.0"
python -m pip install "spconv-cu118==2.3.6" "cumm-cu118==0.4.11"
python -m pip install --no-index --find-links https://data.pyg.org/whl/torch-2.1.0+cu118.html "torch-scatter==2.1.2+pt21cu118"
```

MMDetection3D **1.4.0** was used as an editable installation in the original environment. If it is not already installed, install a compatible MMDetection3D 1.4.0 source checkout. CUDA extensions on Windows may require matching MSVC/CUDA build tools. The provided environment files document the experimental setup; a clean installation has not been independently verified.

## Dataset

The `data/` directory contains the processed XYZRGB maize point clouds and their segmentation annotations. The study used **108 plants**, divided **8:1:1** into **86 training, 11 validation, and 11 test plants**.

Each sample uses the same complete ID across three files:

- `*_pts.npy` — point-cloud features (XYZRGB)
- `*_sem.npy` — semantic labels (`0`: stem; `1`: leaf)
- `*_ins.npy` — instance labels (stem instance `0`; leaves assigned individual instance IDs)

Preserve sample IDs exactly, including suffixes such as `1149-1(2)`. Check the annotation paths and dataset-root settings in `configs/oneformer3d_corn_formal_rgb_full.py` before running training or testing; use the paper's fixed split for evaluation.

## Training

Training **does not require a pretrained checkpoint**:

```powershell
python tools/train.py configs/oneformer3d_corn_formal_rgb_full.py --work-dir work_dirs/gamq_segnet_train
```

The reported experiments used a batch size of **2**, a maximum of **800 epochs**, an initial learning rate of **5e-5**, and a voxel size of **0.01 m**.

## Evaluation

The repository includes the **trained checkpoint selected for testing**, at:

`work_dirs/oneformer3d-full-new/best_all_ap_50%_epoch_660.pth`

This is an **evaluation/inference checkpoint produced by training**, not a pretrained initialization weight and **not required for training from scratch**.

```powershell
python tools/test.py configs/oneformer3d_corn_formal_rgb_full.py "work_dirs/oneformer3d-full-new/best_all_ap_50%_epoch_660.pth" --work-dir work_dirs/gamq_segnet_test
```

Key results reported on the independent test set:

| Metric | GAMQ-SegNet |
|---|---:|
| Stem IoU | 70.98% |
| Leaf IoU | 94.64% |
| mIoU | 82.81% |
| AP | 64.01% |
| PQ | 67.46% |

Results depend on the released evaluation configuration, dataset split, and checkpoint.

## Acknowledgment

This implementation is based on [OneFormer3D](https://github.com/filaPro/oneformer3d): *OneFormer3D: One Transformer for Unified Point Cloud Segmentation* (CVPR 2024). Please also follow the licenses and notices of the underlying libraries.
