# Classification Techniques Comparison – Breast Cancer Dataset

## 🧠 Objective
This notebook showcases three foundational classification algorithms applied to real-world biomedical data:

- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Decision Tree Classifier**

Each model is trained, evaluated, and compared based on its ability to classify malignant vs. benign tumors using Scikit-Learn's Breast Cancer dataset.

---

## 📊 Dataset
- **Source**: `sklearn.datasets.load_breast_cancer()`
- **Features**: 30 numeric features computed from digitized images of breast mass samples
- **Target**: Binary label (malignant = 0, benign = 1)

---

## ✅ Skills Demonstrated
- Data loading and preprocessing (handling nulls, scaling, splitting)
- Supervised learning with scikit-learn
- Model comparison using classification metrics
- Visualization of confusion matrices and decision boundaries
- Practical interpretation of model trade-offs in biomedical contexts

---

## 📘 Notebook Contents
- `breast_cancer_classification_models.ipynb`:
  - Quick EDA and preprocessing
  - KNN, Logistic Regression, and Decision Tree training
  - Comparative evaluation
  - Observations on model performance and real-world implications

---

## 🏁 Results Summary (placeholder)
| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 0.95     | 0.94      | 0.96   | 0.95     |
| Decision Tree      | 0.92     | 0.91      | 0.93   | 0.92     |
| KNN                | 0.93     | 0.92      | 0.93   | 0.92     |

> Logistic Regression continues to punch above its weight. KNN and Decision Trees give solid, explainable backup options.

---

## 🚀 Future Add-ons
- Try Random Forest or XGBoost for more advanced ensemble comparisons
- Deploy a Streamlit app for interactive model selection
