<h1 style="color: skyblue;">
    <b>RepLDM: Reprogramming Pretrained Latent Diffusion Models for High-Quality, High-Efficiency, High-Resolution Image Generation</b>
</h1>

<h2 style="color: thistle;">
    <b>NeurIPS2025 Spotlight ★</b>
</h2>


### 🔥🔥🔥 RepLDM is a training-free method for higher-resolution image generation, enabling the 8k image generation! You can freely adjust the richness of colors and details in the generated image through *attention guidance*.

<!-- here are urls -->
<div align="center">
    <!-- arxiv -->
    <a href="https://openreview.net/pdf?id=QwXpn5IPKk">
        <img src="https://img.shields.io/badge/arXiv-2410.06055-B31B1B.svg" alt="arXiv Paper">
    </a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <!-- project page -->
    <a href='https://kmittle.github.io/project_pages/RepLDM/'>
        <img src='https://img.shields.io/badge/Project-Page-Green'>
    </a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <!-- hugging face -->
    <a href='#'>
        <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'>
    </a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

<!-- authors -->
_**[Boyuan Cao](https://scholar.google.com/citations?user=VBTZ-GMAAAAJ&hl=zh-CN),
[Jiaxin Ye](https://scholar.google.com/citations?hl=zh-CN&user=2VCTS-sAAAAJ&view_op=list_works&sortby=pubdate),
[Yujie Wei](https://weilllllls.github.io/), and
[Hongming Shan*](https://www.hmshan.io/)**_\
(* Corresponding Author)\
From Fudan University
<br>

<img src="fig/teaser.png" width="100%">
</div>
<!-- authors end -->
<!-- urls end -->


## 📝 TODO List
- [ √ ] SDXL: T2I pipeline
- [ √ ] SDXL: T2I & ControlNet pipeline
- [___] SD3: T2I
- [___] FLUX: T2I
- [___] Web UI


## ⚙️ Setup

### Install Environment
```
conda create -n demofusion python=3.9
conda activate demofusion
pip install -r requirements.txt
```


## 🚀 Quik Start

### Quick start with Gradio
**TODO**

### Text to image generation 
**TODO**


## 📖 Overview of RepLDM

### **RepLDM** enables the rapid synthesis of high-quality, high-resolution images **without the need for further training**.

It consists of two stages: 
1. Synthesizing high-quality images at the training resolution using ***Attention Guidance***.
2. Generating finer high-resolution images through pixel upsampling and "diffusion-denoising" loop.

<p align="center">
    <img src="fig/RepLDM.png" width="100%">
</p>

---

### Attention Guidance enables the generation of images with more vivid colors and richer details, as shown in the figure below.

<p align="center">
    <img src="fig/ablation_T2I.png" width="100%">
</p>

---

### Attention Guidance can be used in conjunction with plugins such as ControlNet to achieve an enhanced visual experience, as illustrated in the figure below.

<p align="center">
    <img src="fig/ablation_controlnet.png" width="100%">
</p>

---

### Attention Guidance allows users to freely adjust the level of detail and color richness in an image according to their preferences, simply by modifying the `attention guidance scale`, as shown in the figure below.

<p align="center">
    <img src="fig/attn_guidance_scale_ablation.png" width="100%">
</p>

<!-- **Attention Guidance**: Enhances the structural consistency of the latent representation using a training-free self-attention mechanism. -->

---

### How does attention guidance work?
Attention Guidance computes layout-enhanced representations using a training-free self-attention (TFSA) mechanism and leverages them to strengthen layout consistency:

$\tilde{\boldsymbol{z}} = \gamma\mathrm{TFSA}(\boldsymbol{z})+(1-\gamma) \boldsymbol{z},
\quad 
\mathrm{TFSA}(\boldsymbol{z}) = \mathrm{f}^{-1}\left(\mathrm{Softmax}\left(\frac{\mathrm{f}(\boldsymbol{z}) \mathrm{f}(\boldsymbol{z})^{\mathrm{T}}}{\lambda}\right) \mathrm{f}(\boldsymbol{z})\right),$

where $\boldsymbol{z}$ is the latent representation, $\mathrm{f}$ denotes reshape operation, and 𝛾 and 𝜆 are hyperparameters.
Specifically, Attention Guidance leads each denoising step closer to the final state, as illustrated in the figure below.

<p align="center">
    <img src="fig/attn_guidance_analyze.png" width="100%">
</p>


## 🔬 On Research Comparison
The implementation in the `main branch` includes some modifications based on the original version. If you want to compare with the original method reported in the paper, please refer to the code in the `base branch`.


## 😉 Citation
```
@inproceedings{caorepldm,
  title={RepLDM: Reprogramming Pretrained Latent Diffusion Models for High-Quality, High-Efficiency, High-Resolution Image Generation},
  author={Cao, Boyuan and Ye, Jiaxin and Wei, Yujie and Shan, Hongming},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```
