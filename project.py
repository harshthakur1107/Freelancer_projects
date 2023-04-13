from flask import Flask, render_template, Request
import pickle
import numpy as np

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    Exp = Request.form['Experience']
    Cat = Request.form['Category Name']
    Scat = Request.form['Sub Category Name']
    City = Request.form['Client City']
    Country = Request.form['Client Country']
    data=np.array[Exp,Cat,Scat,City,Country].reshape(-1,1).reshape(-1,1)

    model=pickle.open('modelr.sav','rb')
    Budget=model.predict(data)
    return render_template('predict.html',bud=int(Budget))

@app.route('/predict1', methods=['post','get'])
def predict1():
    Exp = Request.form['Experience']
    Cat = Request.form['Category Name']
    Scat = Request.form['Sub Category Name']
    City = Request.form['Client City']
    Country = Request.form['Client Country']

    model1=pickle.open('modelc.sav','rb')
    Type=model1.predict(np.array[Exp,Cat,Scat,City,Country].reshape(-1,1).reshape(-1,1))
    return render_template('predict1.html',Type=int(Type))


if __name__=='__main__':
    app.run(debug=True)