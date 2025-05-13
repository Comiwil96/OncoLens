import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.linear_model import LogisticRegression

# Load the data to fit RFE
data = pd.read_csv('data/breast_cancer.csv')
X = data.drop(columns=['target'], axis=1)
y = data['target']

# Final model
logreg = LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1', C=0.1)

# Preprocessing
scaler = StandardScaler()
rfe = RFE(estimator=logreg, n_features_to_select=5)
var_thresh = VarianceThreshold(threshold=0.01)

# Build full pipeline
pipeline = Pipeline([
    ('scaler', scaler),
    ('variance', var_thresh),
    ('rfe', rfe),
    ('classifier', logreg)
])

# Fit pipeline to training data
pipeline.fit(X, y)

# Save the pipeline
joblib.dump(pipeline, 'models/logreg_pipeline.pkl')