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
  <a href="https://sandokim.github.io/twings/">
    <img src="https://img.shields.io/badge/Project-Page-blue?style=flat&logo=googlechrome&logoColor=white" alt="Project Page">
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/arXiv-2601.10200-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/Video-YouTube-FF0000?style=flat&logo=youtube&logoColor=white" alt="Video">
  </a>
</p>

<figure align="center">
  <img src="media/fig2.jpg" width="100%">
  <figcaption>
    We introduce <strong>TWINGS</strong>, a framework that enhances 3D Gaussian Splatting (3DGS) by directly addressing point sparsity. We employ Thin Plate Splines (TPS), a smooth non-rigid deformation model that minimizes bending energy to estimate a globally coherent warp from control-point correspondences, to align backprojected points from estimated depth with triangulated 3D control points, yielding calibrated backprojected points. By sampling these calibrated points near the control points, <strong>TWINGS</strong> provides a fast and geometrically accurate initialization for 3DGS, ultimately improving structural detail preservation and color fidelity in reconstructed scenes.
  </figcaption>
</figure>


Official implementation of the CVPR 2026 paper, 
"TWINGS: Thin Plate Splines Warp-aligned Initialization for Sparse-View Gaussian Splatting".


## Code Coming Soon!
ETA: 2026.05 - 2026.06

## Citation
If you find our code or paper helps, please consider citing:
````BibTeX
@inproceedings{hyeseongkim2026twings,
    title = {TWINGS: Thin Plate Splines Warp-aligned Initialization for Sparse-View Gaussian Splatting},
    author = {Hyeseong Kim, Geonhui Son, Deukhee Lee, Dosik Hwang},
    booktitle = {CVPR},
    year = {2026}
}
````


## Contact
Hyeseong Kim (hyseongkim@postech.ac.kr)
