# GAMQ-SegNet: 3D Maize Plant Semantic and Instance Segmentation

GAMQ-SegNet is a point-cloud segmentation framework developed for maize plant structural analysis. It builds on [OneFormer3D](https://github.com/filaPro/oneformer3d) and integrates two proposed components, **GAASE** and **AMSQF**, to support semantic (stem/leaf) and instance-level segmentation of 3D maize plants.

> **Release status:** This is a reproducibility-oriented README draft. Dataset download links, pretrained checkpoint links, and some installation details are pending verification. Replace every `TODO` before making a paper-associated release.

## Highlights

- Semantic segmentation of maize plant point clouds into **stem** and **leaf** classes.
- Instance segmentation for maize plant structural components.
- **GAASE**-enhanced sparse-convolutional backbone (`oneformer3d/spconv_unet.py`).
- **AMSQF**-related query-decoder modifications (`oneformer3d/query_decoder.py`).
- Training, evaluation, and visualization utilities based on the OneFormer3D codebase.

## Repository layout

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
├── Dockerfile
├── LICENSE
├── README.md
└── environment.yml          # To be generated from the verified training environment
```

The `oneformer3d` package name is preserved for compatibility with the original Python import paths.

## Requirements and installation

The upstream OneFormer3D implementation is built on **MMDetection3D v1.1.0**. Its sparse-convolution and CUDA extensions are version-sensitive. Use the tested versions recorded in `environment.yml`, rather than installing the latest PyTorch/MMCV/MMDetection3D packages independently.

```bash
git clone https://github.com/z805947004-ship-it/GAMQ-SegNet.git
cd GAMQ-SegNet
conda env create -f environment.yml
conda activate gamq-segnet
```

Verify that PyTorch detects a CUDA-capable GPU:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

**Important:** The reproducibility environment file must be exported from the environment that actually trained the reported model. Binary packages and custom CUDA operators may require platform-specific installation or compilation. A successful Conda environment creation alone does not guarantee that all CUDA extensions work.

## Dataset

The research dataset contains **108 individual maize plants** represented as 3D point clouds with geometric and color features (XYZRGB). The segmentation tasks use stem/leaf semantic labels and instance annotations as applicable to the released samples.

- Dataset repository / DOI: **TODO — add public or reviewer-accessible dataset record**.
- Training / validation / test sample lists: **TODO — publish the exact splits used in the paper**.
- Label mapping, file formats, units, RGB normalization, and preprocessing steps: **TODO — document against the released dataset**.

Update the dataset root and any annotation or metadata paths in `configs/oneformer3d_corn_formal_rgb_full.py` so they point to the local downloaded dataset. Avoid hard-coded paths from the original development computer.

> Preserve the complete original sample IDs, including suffixes such as `(1)`, `(2)`, or `(3)`, when preparing the published dataset.

## Training

Run the primary GAMQ-SegNet configuration:

```bash
python tools/train.py configs/oneformer3d_corn_formal_rgb_full.py --work-dir work_dirs/gamq_segnet
```

Before training, verify the dataset paths and the configuration's training settings. For paper reproduction, record the random seed, the actual dataset split, the selected checkpoint, and the exact environment versions.

## Evaluation

Evaluate the trained model using the configuration and checkpoint corresponding to the reported experiment:

```bash
python tools/test.py configs/oneformer3d_corn_formal_rgb_full.py \
  /path/to/best_checkpoint.pth \
  --work-dir work_dirs/gamq_segnet_test
```

On Windows PowerShell, run the same command on one line or use PowerShell backticks for line continuation (the backslashes above are for Bash).

For point-cloud prediction/export options, inspect the script's supported command-line arguments before running it:

```bash
python tools/infer_save_ply_corn.py --help
```

The exact prediction export command and quantitative evaluation procedure will be documented after verification against the released code and checkpoint.

## Checkpoints and results

- GAMQ-SegNet pretrained model: **TODO — provide a download URL and checksum**.
- Quantitative results and evaluation protocol: **TODO — add the final paper tables and metric definitions**.
- Baseline and ablation configurations/checkpoints: **TODO — document availability and links**.

Large model weights and the full dataset should be hosted separately (for example, in Zenodo), not committed as ordinary GitHub source files.

## Relationship to OneFormer3D

This project is derived from the [OneFormer3D implementation](https://github.com/filaPro/oneformer3d) associated with:

> Kolodiazhnyi et al., *OneFormer3D: One Transformer for Unified Point Cloud Segmentation*, CVPR 2024. [Paper](https://arxiv.org/abs/2311.14405)

Please also cite OneFormer3D when using code based on the upstream implementation. The upstream software's copyright and licensing terms must be preserved as required. Review the repository's `LICENSE` and the upstream license before assigning terms to newly added source code.

## Citation

If this code supports your research, please cite the GAMQ-SegNet article after publication.

**TODO:** Insert the final paper title, authors, venue, year, DOI, and BibTeX entry when available.

## Contact

Open a GitHub Issue for reproducibility questions. **TODO:** Add an approved research contact address if desired.
