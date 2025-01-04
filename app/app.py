from flask import Flask, render_template, request, jsonify
from utils import model_predict

app = Flask(__name__)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route for web interface
@app.route('/predict', methods=['POST'])
def predict():
    email = request.form.get('content')
    if not email:
        return render_template("index.html", error="No email content provided.")
    
    prediction = model_predict(email)
    return render_template("index.html", prediction=prediction, email=email)

# API endpoint for prediction
@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)  # Parse JSON input
        email = data.get('content')
        if not email:
            return jsonify({'error': 'No email content provided.'}), 400
        
        prediction = model_predict(email)
        return jsonify({'prediction': prediction, 'email': email})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
