import streamlit as st
import requests

st.title("Real-Time Sentiment Analysis")
st.write("Enter a review to predict its sentiment (Positive or Negative).")

review_input = st.text_area("Review Text")

if st.button("Predict"):
    if review_input.strip():
        try:
            # Flask server URL
            api_url = 'http://127.0.0.1:5000/predict'
            
            # Send POST request
            response = requests.post(api_url, json={'review': review_input})
            response_data = response.json()

            if 'error' in response_data:
                st.error(f"Error: {response_data['error']}")
            else:
                st.success("Prediction Results:")
                st.write(f"**Random Forest:** {response_data['RandomForestPrediction']}")
                st.write(f"**XGBoost:** {response_data['XGBoostPrediction']}")
        except Exception as e:
            st.error(f"Error connecting to the backend: {e}")
    else:
        st.warning("Please enter a review text!")
