# DALog-ViT

PyTorch implementation of a post-training quantization (PTQ) framework for
representative vision transformer backbones.
The released ImageNet code supports `vit_*`, `deit_*`, and `swin_*` models at
3-bit and 4-bit quantization settings.

## Getting Started

- Clone this repository.

```
git clone <repository-url> DALog-ViT
cd DALog-ViT
```

- Create a Python environment. The code was developed for CUDA-enabled PyTorch
and uses Python 3.10 or later.

```
conda create -n dalog-vit python=3.10 -y
conda activate dalog-vit
```

- Install PyTorch, torchvision, and the project dependencies.

```
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install timm==0.9.2 numpy pillow tqdm matplotlib
```

## Data Preparation

The released evaluation and experiment scripts use ImageNet-1K. Set
`IMAGENET_DIR` to an ImageNet root containing `train/` and `val/` folders:

```
/path/to/imagenet/
+-- train/
|   +-- n01440764/
|   `-- ...
`-- val/
    +-- n01440764/
    `-- ...
```

```
export IMAGENET_DIR=/path/to/imagenet
```

In Windows PowerShell, use:

```
$env:IMAGENET_DIR = "D:\datasets\imagenet"
```

The ImageNet dataset is not redistributed with this repository. Please obtain
it under the applicable dataset license.

## Pretrained Checkpoints

The evaluation code first looks for checkpoints in
`dalog_haqs_quant/checkpoints/vit_raw/`. For example, the ViT-S checkpoint path
is:

```
dalog_haqs_quant/checkpoints/vit_raw/vit_small_patch16_224.bin
```

If a local checkpoint is absent, the code calls `timm.create_model(..., pretrained=True)` and may download the corresponding pretrained model through
timm. To run offline, place timm-compatible pretrained checkpoints in the
directory above with the model names used by the scripts:

```
vit_tiny_patch16_224.bin
vit_small_patch16_224.bin
vit_base_patch16_224.bin
deit_tiny_patch16_224.bin
deit_small_patch16_224.bin
deit_base_patch16_224.bin
swin_tiny_patch4_window7_224.bin
swin_small_patch4_window7_224.bin
swin_base_patch4_window7_224.bin
```

## Evaluation

Run the commands below from `dalog_haqs_quant/`. The main entry point is
`test_quant.py`; its logs and newly generated checkpoints are written to
`checkpoints/quant_result/<timestamp>/`.

```
cd dalog_haqs_quant
```

### Calibrate a Model 
```
python test_quant.py --model vit_small --config ./configs/3bit/best.py --dataset ~/IMAGENET_DIR --val-batch-size 500 --calibrate
```

### Reconstruction

```
python test_quant.py --model vit_small --config ./configs/3bit/best.py --dataset ~/IMAGENET_DIR --val-batch-size 500 --calibrate --optimize --optim-metric fisher_dplr
```

### Useful Command-Line Arguments

- `--w_bit` and `--a_bit`: weight and activation bit-width.
- `--calib-size`, `--calib-batch-size`, `--optim-size`, `--optim-batch-size`: data configuration for calibration and reconstruction.
- `--val-batch-size`: batch size for validation evaluation.
- `--seed`: random seed.
- `--optim-metric`: reconstruction metric.
- `--optim-mode`: reconstruction input mode.

## License

This project is released under the [Apache-2.0 License](LICENSE).
