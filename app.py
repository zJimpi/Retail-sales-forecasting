import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
sarima_model=pickle.load(open('sarima_monthly_sales_model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    This route will handle both:
    - Form data from HTML frontend
    - JSON data from API calls
    """

    # Check if JSON data is sent
    if request.is_json:
        data = request.get_json()
    else:
        # If form submission from HTML
        data = request.form.to_dict()

    try:
        # Extract number of steps (how many future months/days to forecast)
        steps = int(data.get('steps', 12))  # default: 12 months forecast

        # Forecast using the SARIMA model
        forecast = sarima_model.forecast(steps=steps)

        # Convert to list (for JSON)
        forecast_list = forecast.tolist()

        # Return JSON response
        return jsonify({
            'status': 'success',
            'forecast_steps': steps,
            'predictions': forecast_list
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == "__main__":
    app.run(debug=True)
