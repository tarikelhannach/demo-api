from sklearn.model_selection import cross_val_score
from flask import Flask, request, jsonify
import pandas as pd
import sqlite3
import pickle
import os

os.chdir(os.path.dirname(__file__))


app = Flask(__name__)
app.config['DEBUG'] = True

# WELCOME REQUEST
@app.route("/", methods=['GET'])
def saludo():
    
    return 'Usted se encuentra en la API del modelo de Tarik'

# REQUEST 1
# Create an endpoint that returns the prediction of the new data sent via arguments in the call (/predict):
@app.route('/predict', methods=['GET'])
def predict():

    model = pickle.load(open('data/advertising_model','rb'))

    tv = request.args.get('TV', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)
    prediction = model.predict([[tv,radio,newspaper]])
    return "The prediction of sales investing that amount of money in TV, Radio and Newspaper is: " + str(round(prediction[0],2)) + 'k €'

# REQUEST 2
# An endpoint to store new records in the database that must be previously created. (/ingest_data)

@app.route('/ingest_data', methods=['GET'])
def ingest():

    tv=request.args['TV']
    radio=request.args['radio']
    newspaper=request.args['newspaper']
    sales=request.args['sales']
    
    connection = sqlite3.connect("Data/database.db")
    crsr = connection.cursor()
    query = '''INSERT INTO Advertising VALUES (?,?,?,?)'''
    
    crsr.execute (query,(tv,radio,newspaper,sales)).fetchall()
    connection.commit()
    connection.close()
    
    return "INSERTED."

# REQUEST 3
# Create an endpoint that retrains the model again with the data available in the data folder, that saves that retrained model, returning in the response the mean of the MAE of a cross validation with the new model (/retrain).

@app.route('/retrain', methods=['GET'])
def retrain():

    connection = sqlite3.connect("my_database.db")
    crsr = connection.cursor()

    query = '''SELECT * FROM advertisng'''

    crsr.execute(query)
    data = crsr.fetchall()
    cols = [description[0] for description in crsr.description]
    df = pd.DataFrame(data, columns=cols)

    X = df[['TV', 'radio', 'newsaper']]
    y = df['sales']

    model = pickle.load(open('data/advertising_model','rb'))
    model.fit(X,y)
    scores = cross_val_score(model, X, y, cv=8, scoring='neg_mean_absolute_error')

    pickle.dump(model, open('advertising_model_retrain_v1','wb'))

    return "New model retrained and saved as advertising_model_retrain_v1. The results of MAE with cross validation of 8 folds is: " + str(abs(round(scores.mean(),2)))
app.run()