OncoLens - Breast Cancer Predictor

An end-to-end machine learning project that predicts whether a breast tumor is malignant or benign using histological measurements. This project showcases real-world ML development from exploratory analysis through model optimization and explainability, built using Python, scikit-learn, and SHAP.

## Objective
The goal of this dataset is to explore, train, evaluate and optimize a breast cancer predictor algorithm based on the following models:

- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Decision Tree Classifier**

Each model will be trained, evaluated, and compared based on its ability to classify malignant vs. benign tumors using Scikit-Learn's Breast Cancer dataset. The best performing initial model will be chosen for optimization by means of feature engineering and regularization. The model will then be exported and deployed through a basic UI app.

---

## Features

- Exploratory Data Analysis (EDA)
- Feature Engineering (Correlation filtering, Variance Threshold, RFE)
- Multiple ML Models: Logistic Regression, KNN, Decision Tree
- Hyperparameter Optimization
- SHAP Explainability
- Modular Codebase with `utils.py` and `train.py`
- Optional Streamlit app for interactive prediction
- Cloud-ready model export and deployment (joblib + containerization)

---

## Project Structure
breast-cancer-predictor/
|----app/ # Deployment (Streamlit or Flask app)
|----data/ # Raw and processed datasets
|----models/ # Saved model + scaler artifacts
|----notebooks/ # EDA, prototyping, optimization
|----outputs/ # Visuals, logs, reports
|----src/ # Reusable code (utils, training scripts)
|----requirements.txt # Project Dependencies
|---- README.md # This file

---

## Installation
```bash
# Clone the repo
git clone https://github.com/yourusername/breast-cancer-predictor.git
cd breast-cancer-predictor

# Create virtual env (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

---

## Usage

### Train the model
```bash
python src/train.py

### Run the App (optional)
```bash
streamlit run app/app.y
---

## 📊 Dataset
- **Source**: `sklearn.datasets.load_breast_cancer()`
- **Features**: 30 numeric features computed from digitized images of breast mass samples
- **Target**: Binary label (malignant = 0, benign = 1)

---

## Model Performance
Logistic Regression (Post-Optimization)
- Validation Accuracy: 94%
- Test Accuracy: 96%
Features Used: 5 (selected via Variance Threshold + RFE)

---

## Explainability
SHAP (SHapley Additive exPlanations) is used to interpret model predictions and understand feature impact on classification outcomes

---

## Deployment
The trained model is serialized using joblib and can be deployed via:
- Streamlit for interactive browser-based use
- Flask API with containerization (Docker) for integration into healthcare systems
Cloud platforms(e.g., AWS Lambda, GCP Cloud Run) for production

---

## Author

Cody Wilson (Intuitive Healthcare Technologies)
Built with vision, hustle, and just enough caffeine to hit 97% accuracy.

---

## License
This project is open-source and available under the MIT license