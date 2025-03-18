import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# 🔹 Define dynamic paths for models and encoders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRICE_MODEL_PATH = os.path.join(BASE_DIR, "../models/house_prediction.pkl")
REGION_ENCODER_PATH = os.path.join(BASE_DIR, "../models/region_encoder.pkl")  # For price prediction
RECOMMEND_MODEL_PATH = os.path.join(BASE_DIR, "../models/recommendation_model.pkl")  # KNN Model
LABEL_ENCODERS_PATH = os.path.join(BASE_DIR, "../models/label_encoders.pkl")  # For recommendation model

# 🔹 Load models and encoders
with open(PRICE_MODEL_PATH, "rb") as file:
    price_model = pickle.load(file)

with open(REGION_ENCODER_PATH, "rb") as file:
    region_encoder = pickle.load(file)  # Used for price prediction

with open(RECOMMEND_MODEL_PATH, "rb") as file:
    recommend_model = pickle.load(file)  # Used for recommendations

with open(LABEL_ENCODERS_PATH, "rb") as file:
    label_encoders = pickle.load(file)  # Used for recommendation model (region, locality, etc.)

# 🔹 Load dataset (for recommendations)
dataset_path = os.path.join(BASE_DIR, "../dataset/Mumbai House Prices.csv")
df = pd.read_csv(dataset_path)

# Encode categorical columns for recommendation dataset
for col in ["region", "locality", "type", "status", "age"]:
    df[col] = label_encoders[col].transform(df[col])

# Features used in KNN model
feature_columns = ["region", "locality", "type", "bhk", "status", "age", "area"]
X = df[feature_columns]

# ========================== PRICE PREDICTION API ==========================
@app.route("/predict_price", methods=["POST"])
def predict_price():
    try:
        data = request.json
        region = data.get("region", "").title()
        bhk = data.get("bhk", 0)
        entered_price = float(data.get("user_price", 0))

        # 🔹 Validate inputs
        if not region or bhk <= 0 or entered_price <= 0:
            return jsonify({"error": "Invalid input. Provide valid 'region', 'bhk', and 'user_price'."}), 400

        # 🔹 Encode region for price prediction
        if region not in region_encoder.classes_:
            return jsonify({"error": f"Region '{region}' not found in dataset."}), 400

        region_encoded = region_encoder.transform([region])[0]

        # 🔹 Predict market price
        input_data = np.array([[region_encoded, bhk]])
        predicted_price = price_model.predict(input_data)[0]

        # 🔹 Calculate price variation
        variation = ((entered_price - predicted_price) / predicted_price) * 100
        variation = round(variation, 2)

        # 🔹 Format price output
        predicted_price_formatted = f"₹{predicted_price:.2f} Lakhs"
        if predicted_price >= 100:
            predicted_price_formatted = f"₹{predicted_price/100:.2f} Crores"

        # 🔹 Return response
        return jsonify({
            "predicted_price": predicted_price_formatted,
            "user_price": f"₹{entered_price:.2f} Lakhs",
            "price_variation": f"{variation}% {'above' if variation > 0 else 'below'} market rate"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========================== PROPERTY RECOMMENDATION API ==========================
@app.route("/recommend_properties", methods=["POST"])
def recommend_properties():
    try:
        data = request.json
        searched_regions = data.get("searched_regions", [])

        if not searched_regions or not isinstance(searched_regions, list):
            return jsonify({"error": "Invalid input. Provide an array of searched regions"}), 400

        # 🔹 Convert input to title case
        searched_regions = [region.title() for region in searched_regions]

        # 🔹 Encode searched regions for recommendation
        encoded_regions = []
        for region in searched_regions:
            if region in label_encoders["region"].classes_:
                encoded_regions.append(label_encoders["region"].transform([region])[0])

        if not encoded_regions:
            return jsonify({"error": "No valid regions found in dataset."}), 400

        # 🔹 Filter dataset for searched regions
        filtered_df = df[df["region"].isin(encoded_regions)]

        if filtered_df.empty:
            return jsonify({"message": "No properties found for these regions."}), 200

        # 🔹 Prepare features
        X_filtered = filtered_df[feature_columns]

        # 🔹 Train KNN for the filtered dataset
        knn_filtered = NearestNeighbors(n_neighbors=min(5, len(X_filtered)), metric="euclidean")
        knn_filtered.fit(X_filtered)

        recommended_indices = set()

        # 🔹 Get recommendations for each searched region
        for region in encoded_regions:
            region_properties = filtered_df[filtered_df["region"] == region]
            if region_properties.empty:
                continue

            # Pick a random property
            sample_property = region_properties.sample(1)[feature_columns].values
            _, indices = knn_filtered.kneighbors(sample_property, n_neighbors=5)

            # Store recommendations
            recommended_indices.update(indices[0])

        # 🔹 Get recommended properties
        recommended_df = filtered_df.iloc[list(recommended_indices)].copy()

        # 🔹 Decode categorical columns back
        for col in ["region", "locality", "type", "status", "age"]:
            recommended_df[col] = label_encoders[col].inverse_transform(recommended_df[col])

        # 🔹 Format response
        recommended_properties = recommended_df[["region", "locality", "type", "bhk", "status", "age", "area"]].to_dict(orient="records")

        return jsonify({"recommendations": recommended_properties})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========================== RUN FLASK APP ==========================
if __name__ == "__main__":
    app.run(debug=True)
