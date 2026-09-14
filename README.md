# ABC-GS: Alignment-Based Controllable Style Transfer for 3D Gaussian Splatting

[![arXiv](https://img.shields.io/badge/arXiv-2503.22218-b31b1b.svg)](https://arxiv.org/abs/2503.22218)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://vpx-ecnu.github.io/ABC-GS-website/)

This repository contains the official implementation of the paper **"ABC-GS: Alignment-Based Controllable Style Transfer for 3D Gaussian Splatting"**, introducing a novel approach for style transfer in 3D scenes represented by Gaussian Splatting.

## News
- [2025/11/08] [GT²-GS](https://vpx-ecnu.github.io/GT2-GS-website/) has been accepted by **AAAI 2026**!
- [2025/05/31] Great news! Our ICME 2025 paper has been upgraded to **Oral Presentation**!
- [2025/05/22] We release the homepage for [GT²-GS](https://vpx-ecnu.github.io/GT2-GS-website/), a 3d texture transfer framework for Gaussian Splatting.
- [2025/03/28] We release the paper on arXiv.
- [2025/03/21] Our paper has been accepted by **ICME 2025**!
- [2025/03/26] We release the project page and code of ABC-GS.

## Key Features

- 🎨 **Multi Style Transfer Loss**: Supports four distinct style loss formulations (FAST, NNFM, KNN-FM, Gram Matrix)
- 🌟 **Three-Phase Training**: Pre-processing → Style Transfer → Post-processing pipeline
- 🔍 **Controllable Style Transfer**: Implements senmatic-aware and multi-style for Gaussian scene stylization
- 📦 **Modular Architecture**: Extensible design for custom loss functions and training phases

## Gallery

### Single Style Transfer
![](./abcgs/assets/single_horns.jpg)
![](./abcgs/assets/single_trex.jpg)
![](./abcgs/assets/single_M60.jpg)
![](./abcgs/assets/single_truck.jpg)
### Semantic-aware Style Transfer
![](./abcgs/assets/semantic_flower.jpg)
### Compositional Style Transfer
![](./abcgs/assets/compositional_fern.jpg)


## Installation

### Requirements 
- NVIDIA GPU with CUDA 11.8
- Python 3.10
- PyTorch 2.3.0

### Conda

```bash
# Clone repository with submodules
git clone https://github.com/vpx-ecnu/ABC-GS --recursive
cd ABC-GS

# Install Python dependencies
conda env create -f environment.yaml
conda activate ABC-GS
pip install abcgs/submodules/lang-segment-anything
pip install gs/submodules/diff-gaussian-rasterization
pip install gs/submodules/simple-knn
```



## Quick Start
### Dataset and Checkpoint
* For scene dataset, you can find LLFF dataset in [NeRF](https://github.com/bmild/nerf) and T&T dataset in [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). 
* For style dataset, you can find it in [here](https://drive.google.com/file/d/10EPUQpH0PE8Mnoxxs1URePtjQZElt--s/view?usp=sharing).
* **For optimal stylization results, ensure that the original scene is trained using 0th-order spherical harmonics (SH) coefficients.** Higher-order SH coefficients may introduce artifacts or inconsistencies during the style transfer process. Using 0th-order SH coefficients ensures smoother and more coherent stylization.

### Single Style Transfer
```
python style_transfer.py --config configs/llff_single.yaml
```
### Semantic-aware Style Transfer
```
python style_transfer.py --config configs/llff_semantic.yaml
```
### Compositional Style Transfer
```
python style_transfer.py --config configs/llff_compositional.yaml
```

Please check `python style_transfer.py --help` or files under `configs/` for help.

## Contact

If you have any questions or suggestions, feel free to open an issue on GitHub.
You can also contact [Garv1tum](https://github.com/Grav1tum) and [lzlcs](https://github.com/Lozical) directly.

## Citation

If you find this project useful, please give a star⭐ to this repo and cite our paper:
```bibtex
@INPROCEEDINGS{11209430,
  author={Liu, Wenjie and Liu, Zhongliang and Yang, Xiaoyan and Sha, Man and Li, Yang},
  booktitle={2025 IEEE International Conference on Multimedia and Expo (ICME)}, 
  title={ABC-GS: Alignment-Based Controllable Style Transfer for 3D Gaussian Splatting}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Image segmentation;Three-dimensional displays;Boolean functions;Multimedia systems;Neural radiance field;Data structures;Controllability;Style Transfer;3D Gaussian Splatting;Controllable;Feature Alignment},
  doi={10.1109/ICME59968.2025.11209430}
}

```

## Acknowledgements

This project builds upon the following works:
- **3D Gaussian Splatting**:  
  The core 3D Gaussian rendering and optimization framework is based on the [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) implementation by Bernhard Kerbl et al. 
- **Lang-Segment-Anything**: \
    The semantic segmentation functionality is powered by [Lang-Segment-Anything](https://github.com/luca-medeiros/lang-segment-anything), a language-driven segmentation tool.
