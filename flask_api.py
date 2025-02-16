from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
import flasgger
from flasgger import Swagger

app = Flask(__name__)    
Swagger(app)
pickle_in = open("Classifier.pkl","rb")
classifier = pickle.load(pickle_in)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])

def predict():
    """
    Let's authenticate the note
    there is a change
    
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true

    responses:
        200:
            description: the output values
        

    """
    variance = float(request.form.get('variance'))
    skewness = float(request.form.get('skewness'))
    curtosis = float(request.form.get('curtosis'))  # Fix typo from curtosis to kurtosis
    entropy = float(request.form.get('entropy'))
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    if int(prediction)==0:
        predic = "The note is not real"
    else:
        predic = "The note is real"
    return render_template("index.html", prediction_outcome = "the predicted result is {}".format(predic))

if __name__=="__main__":
    app.run()