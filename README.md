## ___***RepLDM: Reprogramming Pretrained Latent Diffusion Models for High-Quality, High-Efficiency, High-Resolution Image Generation***___

### 🔥🔥🔥 RepLDM is a training-free method for higher-resolution image generation, unlocking the 8k image generation!
<!-- ### 🚀🚀🚀 The ControlNet version and video generation are coming soon! -->

<div align="center">
 <a href='https://openreview.net/pdf?id=QwXpn5IPKk'><img src='https://img.shields.io/badge/OpenReview-Paper-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='#'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='#'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>

_**[Boyuan Cao](https://scholar.google.com/citations?user=VBTZ-GMAAAAJ&hl=zh-CN),
[Jiaxin Ye](https://scholar.google.com/citations?hl=zh-CN&user=2VCTS-sAAAAJ&view_op=list_works&sortby=pubdate),
[Yujie Wei](https://weilllllls.github.io/), and
[Hongming Shan*](https://www.hmshan.io/)**_\
(* Corresponding Author)\
From Fudan University
<br>

<img src="fig/teaser.png" width="100%">
</div>

## 📖 Overview
**RepLDM** enables the rapid synthesis of high-quality, high-resolution images **without the need for retraining**. 

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

## ⚙️ Setup

### Install Environment
```bash
conda create -n apldm python=3.9
conda activate apldm
pip install -r requirements.txt
