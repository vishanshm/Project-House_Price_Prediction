# House Price Prediction (Regression) 🏠

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Wz_bNmv868LcK4Y_P4CD3N5Ml3Iv8Nd2?usp=sharing)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/code/vishanshm/project-house-price-prediction)

## 📌 Project Objective
The goal is to build a robust regression model capable of predicting the continuous variable `SalePrice` based on 79 explanatory variables describing almost every aspect of residential homes.

## 🛠️ Key Workflow & Concepts
* **Target Variable Analysis:** Identified positive skewness in `SalePrice` and applied **Log Transformation** (`np.log1p`) to normalize the distribution for better model performance.
* **Advanced Data Preprocessing:**
    * Handled missing values using context-aware strategies (e.g., filling `LotFrontage` with the median of the neighborhood).
    * Conducted feature encoding using **Label Encoding** for ordinal data and **One-Hot Encoding** for nominal data.
* **Feature Engineering:** Scaled numerical features using `StandardScaler` to ensure uniform influence on the model.
* **Model Comparison:** * Built a baseline **Linear Regression** model.
    * Implemented an advanced **XGBoost Regressor** to handle non-linear relationships and improve accuracy.
* **Evaluation Metrics:** Evaluated models using **RMSE** (Root Mean Squared Error), **MAE** (Mean Absolute Error), and **R-squared**.

## 📊 Results Summary
* **Skewness Correction:** Reduced `SalePrice` skewness from **1.88** to a more normal distribution.
* **Model Performance:** The XGBoost model significantly outperformed the baseline, capturing complex feature interactions that linear models missed.

## 🚀 Repository Structure
```text
Project-House_Price_Prediction/
├── Project_House_Price_Prediction.ipynb   # Full ML Pipeline
├── train.csv                              # Training Data
├── test.csv                               # Testing Data
└── README.md                              # Project documentation
