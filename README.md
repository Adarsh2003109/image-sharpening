# Image Sharpening using knowledge distillation Knowledge Distillation

This repository contains a PyTorch-based implementation of an image deblurring pipeline that uses **Restormer** as a **teacher model** and a lightweight **student model** trained via **knowledge distillation**.

---

## 📁 Project Structure

```
├── restormer.py            # Teacher model definition (Restormer)
├── student_model.py        # Lightweight student model definition
├── train.py                # Training pipeline (this file)
├── motion_deblurring.pth   # Pretrained Restormer weights
├── train_dataset_cache.pkl # Cached training data
├── test_dataset_cache.pkl  # Cached testing data
├── final_student_model.pth # Final trained student model
├── debug/                  # Sample images and visual results
```

---

## 📌 Project Summary

* **Teacher Model**: Restormer (pretrained on motion deblurring)
* **Student Model**: Lightweight U-Net inspired CNN
* **Loss**: Composite of reconstruction, perceptual (VGG16), and feature distillation loss
* **Optimizer**: Adam
* **Scheduler**: ReduceLROnPlateau
* **SSIM Achieved**: \~0.75
* **Mixed Precision**: Enabled
* **Caching**: Datasets are preloaded to RAM and cached to disk for faster iterations

---

## 📂 Dataset

**Source**: [Kaggle - A Curated List of Image Deblurring Datasets](https://www.kaggle.com/datasets/jishnuparayilshibu/a-curated-list-of-image-deblurring-datasets/code)

* Training and testing sets are taken from the **GoPro dataset**.
* Directory structure:

```
data/
├── Gopro/
│   ├── train/
│   │   ├── blur/
│   │   └── sharp/
│   └── test/
│       ├── blur/
│       └── sharp/
```

> ⚠️ Dataset not included in this repo. Please download it separately and update the paths in `Config` class.

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install torch torchvision einops tqdm pillow piq gdown
```

### 2. Download Teacher Weights

Automatically handled by the script from:

```
https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth
```

### 3. Update Paths

Edit the `Config` class to point to your dataset:

```python
class Config:
    TRAIN_BLUR_PATH = "path/to/train/blur"
    TRAIN_SHARP_PATH = "path/to/train/sharp"
    TEST_BLUR_PATH = "path/to/test/blur"
    TEST_SHARP_PATH = "path/to/test/sharp"
```

### 4. Train the Model

```bash
python train.py
```

### 5. Evaluate Model

At the end of training, SSIM and FPS are reported:

```
Final Evaluation:
========================
SSIM: 0.7523
FPS: 32.87 @ (1920, 1080)
========================
```

---

## 🧠 Knowledge Distillation Strategy

The student learns from both the ground truth and the teacher output:

* **Reconstruction Loss**: L1(student, ground\_truth)
* **Perceptual Loss**: L1(VGG(student), VGG(teacher))
* **Feature Distillation**: L1(student, teacher)

Weights are tunable via `alpha`, `beta`, `gamma` in `DistillationLoss`.

---

## 🖼️ Output Samples

Saved in `debug/` folder after training:

* `input_blurry.jpg`
* `output_sharpened.jpg`
* `ground_truth.jpg`

---

## 📦 Deliverables

* [x] Full training code with caching and mixed precision
* [x] Compatible with motion\_deblurring.pth
* [x] SSIM \~0.75 with student model
* [x] Dataset & model path validation

---

## 🏁 Notes

* You can ZIP the repo after training and attach the dataset if needed for offline sharing.
* Avoid pushing the full 6GB dataset to GitHub. Use [Google Drive](https://drive.google.com/) or [Kaggle Datasets](https://www.kaggle.com/datasets) for sharing.

For any issues or enhancements, feel free to open an issue or contribute!
