from flask import Flask, jsonify, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Cargar el modelo de árbol de decisiones
model = pickle.load(open('flores.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Obtener los datos del formulario
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Predecir con el modelo
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Determinar la clase de la flor
        if prediction == 0:
            flower_class = "Setosa"
        elif prediction == 1:
            flower_class = "Versicolor"
        else:
            flower_class = "Virginica"

        return render_template('index.html', prediction_text=f"La clase de la flor es: {flower_class}")
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
