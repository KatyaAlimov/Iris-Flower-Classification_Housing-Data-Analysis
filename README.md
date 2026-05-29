# Iris Flower Classification & Housing Data Analysis

## 📌 Project Overview
This repository focuses on exploratory data analysis (EDA) and dimensionality reduction techniques applied to the classic Iris dataset and structural housing data. The goal is to extract meaningful patterns, evaluate feature distributions, and visualize high-dimensional data using statistical plotting and Principal Component Analysis (PCA).

---

## 📊 Modules & Implementation

### 1. High-Dimensional Visualization via PCA
Reduces feature dimensionality to evaluate structural clusters and separation patterns within the Iris dataset.
*   **Pipeline:** Feature normalization using standard scaling, PCA fitting, and spatial mapping.
*   **Visualizations:** 2D and 3D PCA scatter plots mapped with distinctive class legends.
*   **Analysis:** Evaluation of global variance distribution and cluster overlapping behavior.

### 2. Housing Feature Distribution Analysis
Investigates targeted numerical features within housing transactional data to assess data variance and skewness.
*   **Pipeline:** Data preprocessing, isolation of structural variables (Columns A, K, M, N), and data cleaning.
*   **Visualizations:** Distribution profiling using Boxplots/Violin Plots for statistical spreads, alongside a detailed Histogram tracking Column A.
*   **Analysis:** Assessment of data density, outliers, and internal column characteristics.

---

## 🛠️ Tech Stack & Dependencies
*   **Language:** Python 3.x
*   **Data Manipulation:** `pandas`, `numpy`
*   **Machine Learning:** `scikit-learn` (Standardization & PCA modules)
*   **Data Visualization:** `matplotlib`, `seaborn`

---

## 🚀 Getting Started

### Prerequisites
Ensure you have the required dependencies installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Execution
Place the source datasets (`housing_training.csv` and the Iris dataset) into the root directory and run the pipeline script:
```bash
python AS1.py
```

---

## 📁 Repository Structure
*   `AS1.py` - Core Python application containing pipeline architectures and plotting scripts.
*   `README.md` - Technical project overview and configuration instructions.
*   `.gitignore` - Environment filter preventing cache and system junk tracking.
# Iris_Flower_Classification
