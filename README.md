<h1 style="color: skyblue;">
    <b>RepLDM: Reprogramming Pretrained Latent Diffusion Models for High-Quality, High-Efficiency, High-Resolution Image Generation</b>
</h1>

<h2 style="color: thistle;">
    <b>NeurIPS2025 Spotlight ★</b>
</h2>


### 🔥🔥🔥 RepLDM is a training-free method for higher-resolution image generation, enabling the 8k image generation!

### You can freely adjust the richness of colors and details in the generated image through *attention guidance*.
<!-- ### 🚀🚀🚀 The ControlNet version and video generation are coming soon! -->


<!-- here are urls -->
<div align="center">
    <!-- arxiv -->
    <a href="https://openreview.net/pdf?id=QwXpn5IPKk">
        <img src="https://img.shields.io/badge/arXiv-2410.06055-B31B1B.svg" alt="arXiv Paper">
    </a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <!-- project page -->
    <a href='https://kmittle.github.io/project_pages/RepLDM/'>
        <img src='https://img.shields.io/badge/Project-Page-Green'>
    </a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <!-- hugging face -->
    <a href='#'>
        <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'>
    </a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

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
conda create -n apldm python=3.9
conda activate apldm
pip install -r requirements.txt
```

## 🚀 Quik start

### 🤗 Quick start with Gradio
**TODO**


## 🔬 On Research Comparison
The implementation in the `main branch` includes some modifications based on the original version. If you want to compare with the original method reported in the paper, please refer to the code in the `base branch`.


## 📖 Overview of RepLDM
**RepLDM** enables the rapid synthesis of high-quality, high-resolution images **without the need for further training**. 

It consists of two stages: 
1.  Synthesizing high-quality images at the training resolution using **Attention Guidance**.
2.  Generating finer high-resolution images through pixel upsampling and "diffusion-denoising" loop.

<p align="center">
    <img src="fig/AP-LDM.png" width="90%">
</p>

- **Attention Guidance**: Enhances the structural consistency of the latent representation using a parameter-free self-attention mechanism via linear weighting.
- **Adjustable Detail**: As shown below, increasing the `attn_guidance_scale` results in more details, richer colors, and stronger contrast.

<p align="center">
    <img src="fig/ablation_guidance_scale.png" width="90%">
</p>
