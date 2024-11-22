import logging
import json
import glob
import sys
import os
from flask import Flask
from keras import models
import numpy as np
from flask import request

logging.debug('Init a Flask app')
app = Flask(__name__)

model_dir = "/model"

#model = models.load_model(f"{model_dir}/abalone_model.keras")

def doit(lunghezza, diametro):
    #predict_input = np.array([
    #    [lunghezza,diametro,0.125,0.5095,0.2165,0.1125,0.165,9]
    #])
    #predict_result = model.predict(predict_input)
    
    #test = model.input_shape

    return json.dumps({
        "inputs": predict_input.tolist(),
        "predict_result": predict_result.tolist()
    })

@app.route('/invocations', methods=['POST'])
def invocations():
    data = request.get_json()  # Parse the incoming JSON payload
    lunghezza = data['lunghezza']
    diametro = data['diametro']
    return doit(lunghezza, diametro)
    
@app.route('/ping', methods=['GET'])
def ping():
    try:
        status = 200  # Healthy
    except:
        status = 503  # Service Unavailable

    return '', status