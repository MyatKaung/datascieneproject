from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from src.datascience.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')

@app.route('/train', methods=['GET'])
def training():
    try:
        # Simulate model training process
        os.system("python main.py")
        return jsonify({'status': 'success', 'message': 'Model training completed successfully!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Standardized feature names
        feature_names = [
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
            'density', 'pH', 'sulphates', 'alcohol'
        ]

        # Read inputs from form
        input_data = [
            float(request.form.get(field_name.replace(' ', '_'), 0))
            for field_name in feature_names
        ]

        # Prepare DataFrame for prediction
        input_df = pd.DataFrame([input_data], columns=feature_names)

        # Load prediction pipeline
        obj = PredictionPipeline()
        prediction = obj.predict(input_df)[0]

        # Convert numeric prediction to readable quality
        if prediction < 4:
            quality = "Poor Quality"
        elif prediction < 7:
            quality = "Average Quality"
        else:
            quality = "Good Quality"

        return render_template('result.html', quality=quality, prediction=round(prediction, 2))
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)



# @app.route('/predict', methods=['POST', 'GET']) #route to show the predictions in a web UI
# def index():
#     if request.method == 'POST':
#         try:
#             #reading the inputs given by the user
#             fixed_acidity = float(request.form['fixed_acidity'])
#             volatile_acidity = float(request.form['volatile_acidity'])
#             citric_acid = float(request.form['citric_acid'])
#             residual_sugar = float(request.form['residual_sugar'])
#             chlorides = float(request.form['chlorides'])
#             free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
#             total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
#             density = float(request.form['density'])
#             pH = float(request.form['pH'])
#             sulphates = float(request.form['sulphates'])
#             alcohol = float(request.form['alcohol'])

#             data = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]
#             data = np.array(data).reshape(1, 11)
#             obj = PredictionPipeline()
#             predict = obj.predict(data)
#             return render_template('result.html', prediction=str(predict))
#         except Exception as e:
#             print('The Exception message is: ', e)
#             return 'something is wrong'
#     else:
#         return render_template('index.html')
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


          

