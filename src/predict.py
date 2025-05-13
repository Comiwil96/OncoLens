import pandas as pd
import joblib

# 1. Load the saved pipeline
pipeline = joblib.load("/Users/CodyWilson/Desktop/breast-cancer-predictor/models/logreg_pipeline.pkl")

# 2. Load new data (format must match training X columns)
# Replace with your actual path or data source
new_data = pd.read_csv("data/new_input_data.csv")

# 3. Make predictions
predictions = pipeline.predict(new_data)
probabilities = pipeline.predict_proba(new_data)

# 4. Display or return results
results = pd.DataFrame({
    "Prediction": predictions,
    "Probability_0": probabilities[:, 0],
    "Probability_1": probabilities[:, 1]
})

print(results)

# Optional: Save results to file
results.to_csv("data/predictions_output.csv", index=False)
