"""
Credit Score Classification-Web Application

@author: Arshad Arif
"""
# import necessary libraries
import numpy as np
import pandas as pd
from flask import Flask,request,render_template
import pickle


# create an object app taking current module as argument
app = Flask(__name__)
# load the pickled file 
model = pickle.load(open('xgboost.pkl','rb'))
X_train = pickle.load(open('X_train.pkl','rb'))

#decorator to route to main page
@app.route("/")
def home():
    return render_template('index.html')#returns the home page
    
# decorator to route to prediction page    
@app.route("/predict", methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    final_features = pd.DataFrame(data=[input_features], columns=X_train.columns)
    prediction = model.predict(final_features)
    if prediction == 0:
           result = 'Bad'
    elif prediction == 1:
           result = 'Standard'
    elif prediction == 2:
           result = 'Good'
    
    #returns home page with the prediction
    return render_template('index.html', prediction_text='Your Credit Score is {}!'.format(result))

# to run 
if __name__ == "__main__":
    app.run(debug=True)

