# 📘 Image Reconstruction with PConvNet & Pix2Pix

This repository contains experiments and implementations for **image inpainting and reconstruction** using:

- **Partial Convolutional Networks (PConvNet)**  
- **Pix2Pix (Conditional GAN for Image-to-Image Translation)**  

The project explores **deep learning-based image reconstruction** techniques using **TensorFlow/Keras** and **PyTorch**, focusing on handling missing or corrupted image regions with high fidelity.

---

## 📂 Repository Structure

```
.
├── pconvnet.ipynb                  # Implementation & experiments with Partial Convolutional Networks
├── pconvnet-training-augmented.ipynb # PConvNet training with data augmentation
├── pix2pix-model.ipynb             # Pix2Pix model training & evaluation
└── README.md                       # Project documentation
```

---

## 🚀 Features

- Implementation of **Partial Convolutional Networks** for image inpainting.  
- **Data augmentation** strategies to improve model generalization.  
- **Pix2Pix GAN-based model** for image-to-image translation.  
- Training, evaluation, and visualization of reconstructed images.  

---
## Explination

**1. pconvnet.ipynb**

   Implements a Partial Convolutional Network (PConvNet) for image inpainting.

   Uses partial convolutions where only valid (unmasked) pixels contribute to the convolution operation.

   Demonstrates training and evaluation on masked images to restore missing regions.

   Outputs reconstructed images where holes are filled in with plausible content.

   Goal: Show how PConvNet can handle irregular missing regions better than normal convolution.

**2. pconvnet-training-augmented.ipynb**

   Builds on the first notebook but adds data augmentation.

   Applies transformations like random cropping, flipping, rotations, or noise injection during training.

   Helps the PConvNet generalize better by training on more diverse inputs.

   Shows improved reconstruction quality compared to plain training.

   Goal: Train a more robust PConvNet model with augmented datasets.

**3. pix2pix-model.ipynb**

   Implements the Pix2Pix model (a Conditional GAN).

   Trains a generator + discriminator for image-to-image translation tasks (e.g., converting masked → filled images, or sketches → photos).

   Uses a PatchGAN discriminator to enforce local realism in reconstructed patches.

   Compares GAN-based inpainting against deterministic models like PConvNet.

   Goal: Show how GANs (Pix2Pix) can generate sharper, more realistic textures in inpainting.

## 🛠️ Requirements

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
numpy
pandas
matplotlib
scikit-learn
opencv-python
tensorflow>=2.8
torch
torchvision
tqdm
```

---

## 📊 Usage

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/image-reconstruction.git
   cd image-reconstruction
   ```

2. **Run notebooks**  
   Open Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook
   ```
   - `pconvnet.ipynb` → Train & evaluate **Partial Convolutional Networks**  
   - `pconvnet-training-augmented.ipynb` → Train PConvNet with **augmented dataset**  
   - `pix2pix-model.ipynb` → Run **Pix2Pix GAN** experiments  

3. **Results**  
   Generated reconstructed images and model outputs will be shown inside the notebooks.

---

## 📈 Results

- **PConvNet** achieves robust reconstruction on masked images using partial convolutions.  
- **Pix2Pix** provides realistic texture synthesis but may require more training data.  

---

## 📌 References

- [Liu et al., Image Inpainting for Irregular Holes Using Partial Convolutions (ECCV 2018)](https://arxiv.org/abs/1804.07723)  
- [Isola et al., Image-to-Image Translation with Conditional Adversarial Networks (CVPR 2017)](https://arxiv.org/abs/1611.07004)  

---

## 👨‍💻 Author

Developed by **[Mayank Kumar]([https://github.com/yourusername](https://github.com/Mayank-ac/Deep_Learning-Image-Reconstruction-Challenge))**  
Feel free to raise issues or contribute 🚀

