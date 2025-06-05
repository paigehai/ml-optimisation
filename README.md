# 🕵️ Credit Card Fraud Detection Using Machine Learning

## 📌 Objective

This project explores the detection of fraudulent credit card transactions using five machine learning algorithms:

- Random Forest  
- Support Vector Machine (SVM)  
- XGBoost  
- LightGBM  
- CatBoost  

Each model is evaluated in multiple scenarios, including:
- Baseline (raw imbalanced data)
- After feature transformation
- With class balancing techniques such as **SMOTE**
- With **feature enlargement** and **probability calibration**

The ultimate goal is to identify the most effective model for detecting rare fraudulent transactions, while addressing the challenges of extreme class imbalance.

---

## 📊 Abstract

The dataset (`creditcard.csv`) is heavily imbalanced, with fraudulent transactions representing only **0.17%** of the total. This imbalance biases models toward the majority (non-fraud) class, reducing effectiveness in real-world detection tasks.

### Workflow Summary:

- **Baseline Modeling**: Initial AUC scores ranged from **97.369% (Random Forest)** to **98.621% (SVM)**.
- **Principal Component Analysis (PCA)**: PCA had minimal impact, as the dataset was already compressed. Only 2 features were dropped, so this step was skipped in favour of resampling.
- **Resampling (SMOTE & Undersampling)**: Applied to balance the dataset, improving performance particularly for **Random Forest** and **CatBoost** (up to +1.59% AUC gain).
- **Feature Enlargement**: Added polynomial (squared) terms to increase feature space from 30 to 58. This significantly boosted model performance — e.g., Random Forest AUC jumped from **97.369%** to **98.967%**.
- **Probability Calibration**: Applied to improve predicted probability reliability. Gradient boosting models (CatBoost, XGBoost, LightGBM) benefitted the most. While AUC sometimes decreased slightly, **Brier scores improved**, indicating better calibrated outputs.

---

## 🏁 Results

| Algorithm       | Best AUC Score | Best Configuration                         |
|----------------|----------------|---------------------------------------------|
| CatBoost        | **99.971%**     | SMOTE Resampling                            |
| Random Forest   | 98.967%         | Feature Squaring + Duplicate Removal        |
| SVM             | 98.621%         | Baseline Model                              |
| XGBoost         | 98.405%         | Feature Squaring + Duplicate Removal        |
| LightGBM        | 97.687%         | Feature Squaring + Duplicate Removal        |

> 🏆 **Best Public Test Score**: **99.971% (CatBoost with SMOTE Resampling)**

---

## 📁 Repository Contents

- `mloptimisation.ipynb`: Main notebook containing all experiments and analysis.
- `creditcard.csv`: Input dataset.
- `README.md`: Project overview and methodology.

---

## ⚙️ Tools & Libraries

- Python 3.x
- Scikit-learn
- Imbalanced-learn (for SMOTE)
- XGBoost
- LightGBM
- CatBoost
- Matplotlib (for visualisation)

---

## 📌 Key Takeaways

- **Class imbalance** severely hinders performance; SMOTE and feature engineering can drastically improve results.
- **Feature enlargement** is highly effective for tree-based models.
- **CatBoost** demonstrated the strongest adaptability and generalisation, particularly in handling synthetic data.
- **Probability calibration** improves decision reliability in high-risk applications like fraud detection.
