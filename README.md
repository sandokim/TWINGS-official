<h2 align="center">TWINGS: Thin Plate Splines Warp-aligned Initialization for Sparse-View Gaussian Splatting</h2>

<p align="center">
<a href="https://github.com/sandokim"><strong>Hyeseong Kim</strong></a>
·
<a href=""><strong>Geonhui Son</strong></a>
·
<a href=""><strong>Deukhee Lee</strong></a>
·
<a href=""><strong>Dosik Hwnag</strong></a>
</p>
<h3 align="center">CVPR 2026</h3>


<p align="center">
  <a href="https://sandokim.github.io/twings/"><img src="https://img.shields.io/badge/Project-Page-blue?style=flat&logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2605.22069v2"><img src="https://img.shields.io/badge/arXiv-2605.22069-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://www.youtube.com/watch?v=j7Lb6k3dXCE&t=66s"><img src="https://img.shields.io/badge/Video-YouTube-FF0000?style=flat&logo=youtube&logoColor=white" alt="Video"></a>
</p>


<figure align="center">
  <img src="figures/fig2.jpg" width="100%">
  <figcaption>
    We introduce <strong>TWINGS</strong>, a framework that enhances 3D Gaussian Splatting (3DGS) by directly addressing point sparsity. We employ Thin Plate Splines (TPS), a smooth non-rigid deformation model that minimizes bending energy to estimate a globally coherent warp from control-point correspondences, to align backprojected points from estimated depth with triangulated 3D control points, yielding calibrated backprojected points. By sampling these calibrated points near the control points, <strong>TWINGS</strong> provides a fast and geometrically accurate initialization for 3DGS, ultimately improving structural detail preservation and color fidelity in reconstructed scenes.
  </figcaption>
</figure>

## Generate initial PCD with TWINGS-Init
Please follow the instructions in `TWINGS-Init/README.md` to generate TWINGS-Init. The generated initial pcd will be saved as `{scene}/multiview_pcd/{n_views}/twings_init_pcd.ply`. Use it as `--pcd_path` when training in TWINGS, for example:

```python
--pcd_path nerf_llff_data/fern/multiview_pcd/3_views/twings_init_pcd.ply
```

To use TWINGS-Init as the initialization in other Gaussian Splatting (GS) variants, simply modify the three items below and pass `--pcd_path` to `train.py`.

1) Add a `pcd_path` argument in `arguments/__init__.py`:
```python
class ModelParams(ParamGroup):
    ...
    self._pcd_path = ""
```

2) Use the provided `pcd_path` in `scene/dataset_readers.py`:
```python
def readColmapSceneInfo(path, images, eval, n_views=3, llffhold=8, pcd_path=""):
    ...
    if pcd_path:
        ply_path = pcd_path
```

3) Pass `pcd_path` through the scene loader in `scene/__init__.py`:
```python
scene_info = sceneLoadTypeCallbacks["Colmap"](
    args.source_path,
    args.images,
    args.eval,
    args.n_views,
    args.llff_hold,
    pcd_path=args.pcd_path,
)
```

## TWINGS Installation
```bash
git clone https://github.com/sandokim/TWINGS-official --recursive
cd TWINGS
conda env create --file environment.yaml
conda activate TWINGS
```

## Install submodules
```bash
git submodule update --init --recursive 
pip install ./submodules/simple-knn
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/ml-depth-pro
```

#### Move the `checkpoints` folder of DepthPro to the parent directory of this project. "./checkpoints/depth_pro.pt"
Comment out `ToTensor()` in `submodules/ml-depth_pro/src/depth_pro.py`.
```python
transform = Compose([
    # ToTensor(), 
    Lambda(lambda x: x.to(device)),
    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ConvertImageDtype(precision),
])
```

## Training
`{scene}` is the scene name (e.g., `fern`), and `{n_views}` is the number of training views (e.g., `3`).
```python
python train.py -s {dataset_path}/{scene} -m {exp_name}/{scene} --eval -r 8 --n_views {n_views} --pcd_path {pcd_path}  
# python train.py -s /path/to/nerf_llff_data/fern -m outputs/nerf_llff_data/r8_3v/fern --eval -r 8 --n_views 3 --pcd_path /path/to/nerf_llff_data/fern/multiview_pcd/3_views/twings_init_pcd.ply
```

## Rendering
```python
python render.py -m {exp_name}/{scene} -r 8
# python render.py -m outputs/nerf_llff_data/r8_3v/fern -r 8
```

## Metrics
```python
# LLFF Compute metrics for all scenes
python metric.py --path outputs/nerf_llff_data/r8_3v
# Mip-NeRF360 Compute metrics for all scenes
python metric.py --path outputs/mipnerf360/r8_12v
# DTU Compute masked metrics for all scenes
python metrics_means.py --exp_name outputs/DTU/r4_3v
```

## Scripts
The scripts below run training, rendering, and overall metric mean computation for each dataset in one go.
```bash
bash scripts/dtu.sh
bash scripts/llff.sh
bash scripts/mipnerf360.sh
```

## Download
You can download TWINGS rendered results for LLFF, DTU, and MipNeRF-360, along with TWINGS-Init, from this [link](https://drive.google.com/drive/folders/1p-Zyv71_PWIvOc-JosnD9aW6prQCFl1v?hl=ko).

## Citation
If you find our code or paper helps, please consider citing:
<!-- ````BibTeX
@inproceedings{hyeseongkim2026twings,
    title = {TWINGS: Thin Plate Splines Warp-aligned Initialization for Sparse-View Gaussian Splatting},
    author = {Hyeseong Kim, Geonhui Son, Deukhee Lee, Dosik Hwang},
    booktitle = {CVPR},
    year = {2026}
}
```` -->

````BibTeX
@article{kim2026twings,
  title={TWINGS: Thin Plate Splines Warp-aligned Initialization for Sparse-View Gaussian Splatting},
  author={Kim, Hyeseong and Son, Geonhui and Lee, Deukhee and Hwang, Dosik},
  journal={arXiv preprint arXiv:2605.22069},
  year={2026}
}
````

## Contact
Hyeseong Kim (hyseongkim@yonsei.ac.kr)
