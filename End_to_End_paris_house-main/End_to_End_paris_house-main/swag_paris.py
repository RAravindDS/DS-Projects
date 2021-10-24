from flask import Flask, request
import pickle
import pandas as pd
import numpy as np
from flasgger import Swagger





app = Flask(__name__)
Swagger(app)



pickle_in = open('paris1.pkl','rb')
classifier = pickle.load(pickle_in)


@app.route('/')
def welcome():
    return "Welcome All This is my  First Deployment"


@app.route('/predict')
def predict_paris_regression():


    """ Let's See The predictions
    Based on the inputs you give here.
    ---
    parameters:
      - name: category
        in : query
        type : number
        required : true
      - name: hasYard
        in : query
        type : number
        required : true
      - name: hasPool
        in : query
        type : number
        required : true
      - name: isNewBuilt
        in : query
        type : number
        required : true
      - name: hasStormProtector
        in : query
        type : number
        required : true
      - name: numberOfRooms
        in : query
        type : number
        required : true
      - name: hasStorageRoom
        in : query
        type : number
        required : true
      - name: basement
        in : query
        type : number
        required : true
    responses: 
        200:
            description: The output values

    """ 



























    category = request.args.get('category')
    hasYard = request.args.get('hasYard')
    hasPool = request.args.get('hasPool')
    isNewBuilt = request.args.get('isNewBuilt')
    hasStormProtector = request.args.get('hasStormProtector')
    numberOfRooms = request.args.get('numberOfRooms')
    hasStorageRoom = request.args.get('hasStorageRoom')
    basement = request.args.get('basement')

    prediction = classifier.predict([[category, hasYard, hasPool, isNewBuilt, hasStormProtector, numberOfRooms, hasStorageRoom, basement]])

    return "The Predicted Value is "+ str(prediction)

















if __name__ == '__main__':
    app.run(debug = True)