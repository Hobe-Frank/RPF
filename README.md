<div align="center">

<h1>🎯🌐🔍 Beyond Determinism: Regularized Probabilistic Features for Cross-View Geo-Localization</h1>

[![Code](https://img.shields.io/badge/Code-GitHub-007ACC.svg)](https://github.com/HobeFrank/RPF)

</div>

# Description 📜
--------------

Cross-view geo-localization (CVGL) aims to localize ground images using geotagged satellite imagery. Existing methods project images onto deterministic point features in a shared embedding space, which limits their ability to model real-world image uncertainty arising from factors such as sparse textures and noise. Consequently, the resulting features are often corrupted by unreliable information.

We propose the **Probabilistic CVGL (PCVGL)** paradigm, which extends cross-view representations from point features to probabilistic features for the first time. To instantiate this paradigm, we introduce the **Regularized Probabilistic Features (RPF)** framework. RPF incorporates:

* **Uncertainty Mix Module (UMM)**: Models uncertainty across feature dimensions via cascaded MLPs with residual connections.

* **Uncertainty-Aware Probabilistic Contrastive (UAPC) Loss**: Leverages Bhattacharyya distance to enable variance-driven adaptive feature alignment.

* **Dual Regularization Mechanism (DRM)**: Integrates localization difficulty and feature discriminability into variance learning, consisting of Difficulty-Aware Alignment (DAA) loss and Sparsity Constraint (SC) loss.

# Datasets 📊
We evaluate RPF on three widely used clean benchmarks, and further construct corruption benchmarks to simulate uncertain environments.

## Clean Benchmarks 🏞️

| Dataset | Type | #Pairs | Description |
|:-------:|:----:|:------:|:------------|
| **VIGOR** | Panorama ↔ Satellite | 105k / 90k | 4 US cities, cross-area & same-area modes |
| **CVACT** | Panorama ↔ Satellite | 35.5k / 8.9k | Suburban & rural regions of Canberra |
| **University-1652** | Drone ↔ Satellite | 1,652 buildings | 72 universities worldwide |

## Corruption Benchmarks 🔧

To evaluate robustness, we introduce corruptions (blur, noise, weather, digital) at multiple severity levels on VIGOR, CVACT, and University-1652, following the protocol of CVACT_val-C.

#Framework Architecture 🖇️

<div align="center">
  <img src="figures/framework.jpg" alt="RPF Framework" width="800">
  <br>
  <em>Overall framework of RPF. A dual-branch Siamese network encodes images as Gaussian distributions (μ, Σ). UMM extracts variance, UAPC loss uses Bhattacharyya distance, and DRM regularizes uncertainty with difficulty alignment.</em>
</div>

## Key Features ✨

### Probabilistic Feature Representation
- Maps each image to a Gaussian distribution: \( p(f|I) = \mathcal{N}(f; \mu, \text{diag}(\Sigma)) \)
- Mean μ captures semantic content, variance Σ models dimension-wise reliability

### Uncertainty Mix Module (UMM)
- Flattens patch features and performs multi-layer feature mixing
- Preserves amplitude information (no normalization) to retain scale cues for uncertainty
- Outputs non-negative variance via Softplus activation

### Uncertainty-Aware Probabilistic Contrastive Loss
- Uses Bhattacharyya distance (BD) to measure distribution overlap
- BD adaptively weights mean differences by inverse variance, suppressing unreliable dimensions
- Encourages variance structure consistency across cross-view pairs

### Dual Regularization Mechanism (DRM)
- **DAA Loss**: Aligns variance with localization difficulty (similarity gap between positive and hardest negative)
- **SC Loss**: Sparsity constraints on both variance and difficulty signal to prevent variance inflation and enhance feature separability

 # Performance Results 🚀

## Performance on Clean Benchmarks

### VIGOR Dataset (Cross-Area / Same-Area)

**Table: Comparison with SOTA methods on the VIGOR dataset. The best results are highlighted in bold.**

| Method               | Publication  | Params (M) | Flops (G) | Cross-Area (R@1↑ / R@5↑ / HI↑) | Same-Area (R@1↑ / R@5↑ / HI↑) |
|:--------------------:|:------------:|:----------:|:---------:|:------------------------------:|:-----------------------------:|
| PaSS-KD              | TCVST’ 2024  | –          | –         | 21.00 / 39.70 / 22.20          | 52.90 / 76.60 / 57.00         |
| SCPNet               | TCVST’ 2025  | –          | –         | 34.42 / 57.47 / 38.12          | 66.89 / 90.01 / 75.88         |
| FRGeo                | AAAI’ 2024   | **27.80**  | **18.18** | 37.54 / 59.58 / 40.66          | 71.26 / 91.38 / 82.41         |
| Sample4Geo           | ICCV’ 2023   | 87.51      | 45.12     | 61.70 / 83.50 / 69.87          | 77.86 / 95.66 / 89.82         |
| AuxGeo               | ISPRS’ 2025  | 87.51      | 45.12     | 63.94 / 84.98 / 76.25          | **80.34** / 96.25 / **93.78** |
| CV-cities            | JSTARS’ 2024 | 90.50      | 91.63     | 64.61 / 87.48 / 75.97          | 78.27 / 96.10 / 90.76         |
| CV-cities †          | ECCV’ 2024   | 90.50      | 91.63     | 65.52 / 88.23 / 76.94          | 76.82 / 95.76 / 89.64         |
| **RPF (ours)**       | –            | **95.49**  | **95.66** | **68.84** / **89.78** / **80.15** | 79.83 / **97.14** / 93.03    |

### University-1652 Dataset

**Table: Comparison with SOTA methods on the University-1652 dataset.**

| Method               | Publication  | Satellite→Drone (R@1↑ / AP↑) | Drone→Satellite (R@1↑ / AP↑) |
|:--------------------:|:------------:|:----------------------------:|:----------------------------:|
| Sample4Geo           | ICCV’ 2023   | 95.14 / 91.39                | 92.65 / 93.81                |
| CCR                  | TCVST’ 2024  | 95.15 / 91.80                | 92.54 / 93.78                |
| Ge et al.            | TGRS’ 2024   | 95.58 / 92.17                | 92.79 / 93.91                |
| CV-cities            | JSTARS’ 2024 | 96.01 / 92.57                | 97.43 / 95.01                |
| CV-cities †          | ECCV’ 2024   | 95.99 / 95.60                | 97.51 / 95.49                |
| MEAN                 | TCVST’ 2025  | 96.01 / 92.08                | 93.55 / 94.53                |
| SHAA                 | TCVST’ 2025  | 96.15 / 93.49                | 93.69 / 94.68                |
| CDM-Net              | TGRS’ 2025   | 96.68 / 94.05                | 95.13 / 96.04                |
| **RPF (ours)**       | –            | **96.77** / **95.93**        | **97.72** / **95.93**        |

## Robustness on Corruption Benchmarks

### University‑1652_val‑C

**Table: Comparison with SOTA methods on the corrupted University-1652_val-C dataset (Satellite→Drone).**

| Method           | Blur-Gaussian  <br>(R@1_c↑ / RCE_c↓) | Weather-Brightness  <br>(R@1_c↑ / RCE_c↓) | Noise-Shot  <br>(R@1_c↑ / RCE_c↓) | Corruption Avg.  <br>(R@1_c↑ / RCE_c↓) |
|:----------------:|:-------------------------------:|:-------------------------------------:|:----------------------------:|:----------------------------------:|
| MEAN             | 79.17 / 16.92                   | 91.14 / 4.36                         | 86.45 / 9.28                 | 85.59 / 10.19                     |
| CCR              | 78.84 / 17.39                   | 91.26 / 4.38                         | 86.78 / 9.07                 | 85.63 / 10.28                     |
| Sample4Geo       | 84.17 / 10.47                   | 90.14 / 4.12                         | 86.21 / 8.30                 | 86.84 / 7.63                      |
| CV-cities        | 87.58 / 8.76                    | 90.06 / 6.18                         | 85.82 / 10.60                | 87.82 / 8.51                      |
| CV-cities †      | 87.74 / 8.59                    | 91.60 / 4.18                         | 87.44 / 8.91                 | 89.05 / 7.23                      |
| **RPF (ours)**   | **90.33** / **6.66**            | **93.04** / **3.85**                 | **88.95** / **8.08**         | **90.77** / **6.20**              |


**Table: Comparison with SOTA methods on the corrupted University-1652_val-C dataset (Drone→Satellite).**

| Method           | Blur-Gaussian  <br>(R@1_c↑ / RCE_c↓) | Weather-Brightness  <br>(R@1_c↑ / RCE_c↓) | Noise-Shot  <br>(R@1_c↑ / RCE_c↓) | Corruption Avg.  <br>(R@1_c↑ / RCE_c↓) |
|:----------------:|:-------------------------------:|:-------------------------------------:|:----------------------------:|:----------------------------------:|
| CCR              | 54.80 / 40.94                   | 88.41 / 4.71                         | 77.93 / 16.01                | 73.71 / 20.55                     |
| MEAN             | 60.31 / 35.01                   | 87.58 / 5.62                         | 78.06 / 15.88                | 75.32 / 18.84                     |
| Sample4Geo       | 66.97 / 27.67                   | 86.96 / 6.09                         | 77.94 / 15.83                | 77.29 / 16.53                     |
| CV-cities        | 92.58 / 4.98                    | 94.72 / 2.78                         | 92.49 / 5.07                 | 93.26 / 4.28                      |
| CV-cities †      | 93.44 / 4.24                    | 95.72 / 1.90                         | 93.58 / 4.09                 | 94.25 / 3.41                      |
| **RPF (ours)**   | **95.24** / **2.53**            | **96.39** / **1.36**                 | **93.96** / **3.84**         | **95.20** / **2.58**              |

### VIGOR_val‑C

**Table: Comparison with SOTA methods on the corrupted VIGOR_val-C dataset. The table reports the results of different methods for corruption Avg.**

| Method           | Same-Area (R@1_c↑ / RCE_c↓) | Cross-Area (R@1_c↑ / RCE_c↓) |
|:----------------:|:---------------------------:|:----------------------------:|
| FRGeo            | 29.87 / 58.09               | 11.26 / 69.99                |
| Sample4Geo       | 35.43 / 53.77               | 23.94 / 60.15                |
| AuxGeo           | 37.57 / 52.65               | 22.93 / 63.27                |
| CV-cities        | 52.04 / 33.52               | 34.54 / 46.54                |
| CV-cities †      | 50.95 / 33.68               | 36.24 / 44.69                |
| **RPF (ours)**   | **54.87** / **31.26**       | **39.16** / **43.11**        |

## Understanding Uncertainty 🔍

We demonstrate that uncertainty in RPF is not merely noise but a meaningful indicator of localization difficulty:
- Higher uncertainty correlates with increased similarity between positive and hardest negative samples.
- Removing high-uncertainty samples substantially boosts R@1 (from 68.4% → 86.2% on clean data).
- Variance features suppress uncertain regions (e.g., roads) allowing mean features to focus on discriminative structures (e.g., buildings).

<div align="center">
  <img src="figures/uncertainty_analysis.jpg" alt="Uncertainty analysis" width="700">
  <br>
  <em>(a) Comparison of uncertainty score distributions. (b) Sparsification curve: R@1 vs. the ratio of high-uncertainty samples removed. (c) and (d) relationships between uncertainty quantiles and R@1, and relative localization difficulty, respectively. Shaded areas indicate the 95% confidence interval. Clean and Corruption correspond to VIGOR Cross-Area and VIGOR_val-C Cross-Area results.</em>
</div>

## Installation & Usage 🚂

### Requirements

```bash
pip install -r requirements.txt
```

### Training

```bash
cd rpf/train
python train_universitySD.py
```
## Code Availability 📦

The remaining code will be released upon paper acceptance.

## Acknowledgements 🙏

This work is supported by the Natural Science Foundation of China. We thank the authors of:
- [CV-cities](https://github.com/GaoShuang98/CVCities)
- [Sample4Geo](https://github.com/Skyy93/Sample4Geo)
---

<div align="center">
  <strong>🌟 Star us on GitHub if you find this project helpful! 🌟</strong>
</div>
