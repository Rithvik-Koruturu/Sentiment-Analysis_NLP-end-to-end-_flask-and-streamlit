from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load models and preprocessor
cv = pickle.load(open('coutVectorizer.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
rf_model = pickle.load(open('random_forest_model.pkl', 'rb'))  # Random Forest Model
xgb_model = pickle.load(open('xgb_model.pkl', 'rb'))          # XGBoost Model

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        review = data['review']

        # Preprocess the review
        cleaned_review = re.sub('[^a-zA-Z]', ' ', review).lower().split()
        ps = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        processed_review = ' '.join([ps.stem(word) for word in cleaned_review if word not in stop_words])
        
        # Vectorize and scale
        vectorized_review = cv.transform([processed_review]).toarray()
        scaled_review = scaler.transform(vectorized_review)

        # Predict using Random Forest
        rf_prediction = rf_model.predict(scaled_review)
        xgb_prediction = xgb_model.predict(scaled_review)

        # Response
        result = {
            'RandomForestPrediction': 'Positive' if rf_prediction[0] == 1 else 'Negative',
            'XGBoostPrediction': 'Positive' if xgb_prediction[0] == 1 else 'Negative'
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
