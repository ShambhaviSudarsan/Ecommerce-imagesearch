import numpy as np
from flask import Blueprint,Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import LabelEncoder

with open("model-ecommerce-lasso.pkl", 'rb') as file:  
    model = pickle.load(file)

model_predict = Blueprint('model_predictor',__name__)

@model_predict.route('/predict_api',methods=['POST'])
def predict_api():

    data = request.get_json(force=True)
    datalist=list(data.values())

    for i in range(len(datalist)):
        datalist[i]=int(datalist[i])
    
    prediction = model.predict([np.array(datalist)])

    output = int(round(prediction[0]))
   
    return jsonify(output)
