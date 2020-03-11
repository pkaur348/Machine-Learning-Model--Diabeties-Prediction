# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 21:22:50 2019

127:0.0.1:5000/predict?Pregnancies=2&Glucose=123&BloodPressure=70&SkinThickness=32&Insulin=25&BMI=33.7&DiabetesPedigreeFunction=0.34&Age=24

"""

from flask import Flask, request

from sklearn.externals import joblib

app = Flask(__name__)

app.config['SECRET_KEY'] = '185552ee79b95239f18ed7c222d3ca5f'

@app.route('/predict')
def predict():
    Pregnancies              = float(request.args['Pregnancies']) #if key doesn't exist, returns a 400, bad request error
    Glucose                  = float(request.args['Glucose'])
    BloodPressure            = float(request.args['BloodPressure'])
    SkinThickness            = float(request.args['SkinThickness'])
    Insulin                  = float(request.args['Insulin'])
    BMI                      = float(request.args['BMI'])
    DiabetesPedigreeFunction = float(request.args['DiabetesPedigreeFunction'])
    Age                      = float(request.args['Age'])
    
    classifier               = joblib.load('predict.result')
    result                   = classifier.predict([[Pregnancies, Glucose, BloodPressure,
                                                   SkinThickness, Insulin, BMI, 
                                                   DiabetesPedigreeFunction, Age]])
    
    
    return str(result[0])

if __name__ == '__main__':
    app.run(debug=True)