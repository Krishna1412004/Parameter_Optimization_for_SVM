# Parameter_Optimization_for_SVM
# 🍷 Wine Dataset Classification using NuSVC and Hyperparameter Tuning

This project applies a Support Vector Machine (SVM) classifier using the `NuSVC` model to the [Wine dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html). The goal is to evaluate the effect of different SVM kernels and hyperparameters (`nu`, `gamma`) across multiple random samples.

---

## 📊 Dataset Overview

- **Samples:** 178  
- **Features:** 13  
- **Target Classes:** 3 (Class 0, 1, 2)  
- **Class Distribution:**
  - Class 1: 71 samples
  - Class 0: 59 samples
  - Class 2: 48 samples

Noise was added and features were scaled for better model generalization.

---

## ⚙️ Methodology

- Applied `NuSVC` model with 4 kernel types: `linear`, `poly`, `rbf`, `sigmoid`
- Randomized `nu` and `gamma` values to simulate hyperparameter tuning
- Executed 100 random trials for each of the 10 samples
- Recorded the best accuracy and corresponding parameters per sample
- Visualized the convergence (accuracy vs iteration) of the best-performing sample

---

## 🏆 Results Summary

| Sample | Best Accuracy (%) | Best SVM Parameters                              |
|--------|-------------------|--------------------------------------------------|
| S1     | 98.15             | Kernel: linear, Nu: 0.54, Gamma: scale           |
| S2     | 98.15             | Kernel: poly, Nu: 0.12, Gamma: 0.4               |
| S3     | 96.30             | Kernel: poly, Nu: 0.1, Gamma: 0.439              |
| S4     | **100.00**        | Kernel: rbf, Nu: 0.2, Gamma: 0.071              |
| S5     | 98.15             | Kernel: linear, Nu: 0.31, Gamma: scale           |
| S6     | **100.00**        | Kernel: rbf, Nu: 0.54, Gamma: 0.084             |
| S7     | **100.00**        | Kernel: linear, Nu: 0.17, Gamma: scale           |
| S8     | 98.15             | Kernel: rbf, Nu: 0.14, Gamma: 0.106             |
| S9     | 96.30             | Kernel: linear, Nu: 0.2, Gamma: scale            |
| S10    | **100.00**        | Kernel: rbf, Nu: 0.14, Gamma: 0.012             |

🔹 **Best Sample:** S4  
🔹 **Best Accuracy:** 100.00%  
🔹 **Convergence Plot**: ![Accuracy vs Iteration](convergence_plot.png)

---

## 📌 Requirements

- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib

Install using:

```bash
pip install numpy pandas matplotlib scikit-learn
