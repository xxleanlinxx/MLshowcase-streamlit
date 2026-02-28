# 🕹️ Machine Learning & XAI Demo App

## 👁️‍🗨️ Overview

- This [Demo application](https://ml-xai-showcase-toolkit.streamlit.app/) demonstrates the integration of **Machine Learning** techniques and **Explainable AI (XAI)** methods using Streamlit.
- It utilizes Seaborn datasets (`mpg` and `titanic`) and the **SECOM** semiconductor manufacturing dataset as examples for regression and classification tasks.
- Key functionalities include *dataset exploration*, *statistical analysis*, *model summary*, *feature importance visualization*, and *partial dependence plots*.

---

## 📎 Features

- **Dataset Selection**
  > - Choose between preloaded datasets (`mpg`, `titanic`, and `secom`).
- **Dataset Summary**
  > - View column descriptions, data types, and statistics.
- **Exploratory Data Analysis (EDA)**:
  > - ANOVA(`one-way` & `three-way`)
  > - Visualize data distributions
  > - Multi-Collinearity diagnosis(`VIF`)
  > - Correlation analysis(`Correlation Matrix` & `Pair plot`)
- **Machine Learning Models**
  > - Regression case with `LGBMRegressor` on the `mpg` dataset.
  > - Classification case with `RandomForestClassifier` on the `titanic` dataset.
  > - **Classification case with `XGBoost` on the `secom` dataset** — featuring `Logistic Regression` as baseline, `SMOTE` for class imbalance, threshold tuning via Youden's J, and comprehensive feature engineering (high-missing removal, zero-variance removal, high-correlation filtering).
- **Explainable AI**(XAI)
  > - SHAP summary plots for feature importance.
  > - 2-Dimensional Partial dependence plots(PDP) for interaction effects.
  > - SHAP waterfall plots for individual predictions.
  > - **Null Importance Assessment** for validating feature significance (SECOM).

---

## 📂 File Structure

```plaintext
ml-xai-demo/
├── main.py                          # Main Streamlit app file
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore (excludes raw SECOM data)
├── README.md                        # Documentation
├── SECOM_Report_EN.md               # SECOM detailed report (English)
├── SECOM_Report_ZH.md               # SECOM detailed report (中文)
└── assets/
    ├── train_mpg.py                 # MPG model training script
    ├── train_titanic.py             # Titanic model training script
    ├── train_secom.py               # SECOM model training script
    ├── mpg_*.pkl / .npy             # Pre-computed MPG model artifacts
    ├── titanic_*.pkl / .npy         # Pre-computed Titanic model artifacts
    ├── secom_*.pkl / .npy           # Pre-computed SECOM model artifacts
    ├── secom/                       # Raw SECOM data (gitignored)
    └── *.png                        # App images & icons
```

---

## ⚡ Dependencies

This application uses the following Python libraries:

- **Basic**: `streamlit`, `pickle`, `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`, `lightgbm`, `xgboost`, `imbalanced-learn`
- **Explainable AI**: `shap`, `pdpbox`
- **Statistics**: `statsmodels`, `scipy`

Install all dependencies via `requirements.txt`.

---

## 🏭 SECOM Dataset

The **SECOM** (Semiconductor Manufacturing) dataset is a classic industrial ML benchmark:

| Property | Value |
|---|---|
| Samples | 1,567 |
| Raw Features | 590 sensor/process measurements |
| Features after cleaning | ~272 |
| Target | Pass (-1) / Fail (1) binary classification |
| Class Imbalance | ~14:1 (1,463 Pass vs 104 Fail) |

**Pipeline:** Feature Engineering → SMOTE → StandardScaler → Logistic Regression (baseline) + XGBoost (main) → SHAP + Null Importance

> See `SECOM_Report_EN.md` / `SECOM_Report_ZH.md` for detailed analysis.

---

## 📷 Screenshots

### Home Page
![Home Page](assets/home_page.png)

### SHAP Summary Plot
![SHAP Summary Plot](assets/shap_summary.png)

### SHAP WaterFall Plot
![SHAP WaterFall Plot](assets/shap_waterfall.png)

---

## 📃 Contributing

Feel free to open issues or submit pull requests for improvements. Contributions are welcome!

---

## 🧰 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

### 👾 Author
Developed with ❤️ by [Lean Lin]. 

For any queries or suggestions, please contact:
- [Gmail](mailto:xphoenixx32@gmail.com)
- [LinkedIn](https://www.linkedin.com/in/leanlin/)
