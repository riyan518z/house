from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
try:
    with open('house_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input values from the form
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        flat_area = float(request.form['flat_area'])
        lot_area = float(request.form['lot_area'])
        floors = float(request.form['floors'])
        waterfront = float(request.form['waterfront'])
        condition = float(request.form['condition'])
        grade = float(request.form['grade'])
        age = float(request.form['age'])

        # Arrange features in training order
        features = [
            bedrooms, bathrooms, flat_area, lot_area,
            floors, waterfront, condition, grade, age
        ]

        # Convert to numpy array and reshape
        final_features = np.array(features).reshape(1, -1)

        # Make prediction
        if model:
            prediction = model.predict(final_features)
            output = round(prediction[0], 2)
            return render_template(
                'index.html',
                prediction_text=f'🏡 Predicted Sale Price: ₹{output:,.2f}'
            )
        else:
            return render_template(
                'index.html',
                prediction_text='❌ Model not loaded.'
            )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f'❌ Error: {str(e)}'
        )

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)