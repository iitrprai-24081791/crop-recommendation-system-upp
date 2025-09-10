import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Train model
X = df.drop("label", axis=1)
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🌱 Crop Recommendation System")
st.write("Enter soil and weather details to get the best crop recommendation.")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
temperature = st.number_input("Temperature (°C)", min_value=0, max_value=50, value=25)
humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
ph = st.number_input("pH value", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0, max_value=500, value=100)

# Prediction with Top-3 recommendations
if st.button("Recommend Crop"):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Prediction probabilities
    probs = model.predict_proba(features)[0]
    
    # Get top 3 crop indices
    top3_idx = np.argsort(probs)[-3:][::-1]
    
    st.success(f"🌾 Best Recommended Crop: **{model.classes_[top3_idx[0]]}**")
    
    st.write("### 🌟 Other Possible Crops:")
    for i in top3_idx:
        st.write(f"- {model.classes_[i]} → {probs[i]*100:.2f}% confidence")
