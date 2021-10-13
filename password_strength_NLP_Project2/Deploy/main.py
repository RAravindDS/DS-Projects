from flask import Flask, request
import pickle
import pandas as pd
import numpy as np
from flasgger import Swagger





app = Flask(__name__)
Swagger(app)



pickle_in = open('xgb.pkl','rb')
classifier = pickle.load(pickle_in)

pickle_out = open('xgbtranss.pkl', 'rb')
classifier1 = pickle.load(pickle_out)

@app.route('/')
def welcome():
    return "Welcome All This is my  First Deployment"


@app.route('/predict')
def predict_paris_regression():


    """ Let's See The predictions of your passwords
    Based on the inputs you give here.
    ---
    parameters:
      - name: password
        in : query
        required : true
    one of:
        type : string
        type : integer
        
      
      
    responses: 
        200:
            description: The output values

    """ 




   
    password = request.args.get('password')
    passwords = classifier1.transform([password])

    

    prediction = classifier.predict(passwords)

    return "The Predicted Value is "+ str(prediction)

















if __name__ == '__main__':
    app.run(debug = True)


