# Parallel qMRI Reconstruction from 4x Accelerated Acquisitions 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Paper 
- **Parallel qMRI Reconstruction from 4x Accelerated Acquisitions**, Mingi Kang [[arXiv](https://arxiv.org/abs/2511.18232)] 

## Funding & Acknowledgements
This project was supported by the Washington University in St. Louis Undergraduate Summer Engineering Research Fellowship (WUSEF) 2025. [[WUSEF Website](https://engineering.washu.edu/academics/undergraduate-research/WUSEF.html)]

Project was under the supervision of Prof. Ulugbek S. Kamilov at Washington University in St. Louis (now at University of Wisconsin-Madison) and the Computational Imaging Group (CIG) lab. 

**Project period:** Summer 2025

## Overview
Magnetic Resonance Imaging (MRI) acquisitions require extensive scan times, limiting patient throughput and increasing susceptibility to motion artifacts. This project introduces a miniature version of **SPICER: self-supervised learning for MRI with automatic coil sensitivity estimation and reconstruction**. The proposed model effectively reduces the parameter size by 4x of deep unfolding network utilized in SPICER while maintaining reconstruction performance. Our two-module architecture consists of Coil Sensitivity Map (CSM) estimation modeuls and a U-Net-based MRI reconstruction module. 

### Models 
1. `unet.py` - 
    - Implementation of the standard U-Net architecture for image segmentation tasks.

2. `attention_unet.py` -
    - Implementation of the Attention U-Net architecture, which incorporates attention gates to enhance feature learning.

3. `deep_unfolding.py` - 
    - Implementation of a deep unfolding network for MRI reconstruction, combining model-based and data-driven approaches. Implementations from SPICER paper. 

4. `recon_unet.py` - 
    - Implementation of a U-Net-based MRI reconstruction model, designed for efficient and accurate image reconstruction.

5. `recon_attention_unet.py` - 
    - Implementation of an Attention U-Net-based MRI reconstruction model, enhancing the standard U-Net with attention gates for improved performance.

## Project Structure

```
.
├── models/
│   ├── unet.py             # U-Net architecture implementation
│   ├── attention_unet.py   # Attention U-Net architecture implementation
│   ├── deep_unfolding.py   # Deep unfolding network implementation
│   ├── recon_unet.py       # U-Net-based MRI reconstruction model
│   └── recon_attention_unet.py # Attention U-Net-based MRI reconstruction model
├── utils/
│   ├── fourier.py          # Fourier transform utilities
│   ├── loss.py             # Loss functions for training
│   ├── metric.py           # Evaluation metrics for MRI reconstruction
│   ├── operator.py         # MRI reconstruction operators
│   └── util.py             # Utility functions
├── dataset.py              # Multi-coil Echo MRI dataset
├── main.py                 # Main training script
├── test.py                 # Testing script for evaluation
├── visualize.py            # Visualization functions (save images, videos)
├── README.md               # ← you are here
└── LICENSE                 # MIT License
```

## License
MRI-Reconstruction is released under the MIT License. Please see the [LICENSE](https://www.google.com/search?q=LICENSE) file for more information.


