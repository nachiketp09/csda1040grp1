import pandas as pd
import numpy as np
import pickle

import dash
from flask import Flask, render_template, request
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

import io
import warnings
warnings.filterwarnings('ignore')
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# Load the pickle files of the models and CountVectorizer
cv1 = pickle.load(open('cv1.pkl', 'rb'))
rfmodel = pickle.load(open('rfmodel.pkl', 'rb'))
rfmodel = pickle.load(open('lgmodel.pkl', 'rb'))
mnbmodel = pickle.load(open('mnbmodel.pkl', 'rb'))


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

    
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        
        message = message.lower()
        message_words = message.split()
        message_words = [word for word in message_words if not word in set(stopwords.words('english'))]
        ps = PorterStemmer()
        final_review = [ps.stem(word) for word in message_words]
        final_review = ' '.join(final_review)

        temp = cv1.transform([final_review]).toarray()
        prediction = mnbmodel.predict(temp)
        
        #vect = cv1.transform(data).toarray()
        #prediction = mnbmodel.predict(vect)
        
        return render_template('result.html', prediction=prediction)
        

if __name__ == '__main__':
    app.run_server(debug=True)