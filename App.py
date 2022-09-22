from flask import Flask, render_template, request, app, jsonify, url_for
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("Reg_Model.pkl", "rb"))
scalar = pickle.load(open("Scaler_Pickle.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict-api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    print(new_data)
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    output = round(output,2)
    return render_template("home.html",prediction_text="The House Price is estimated to be {}".format(output))
    

if __name__ == '__main__':
    app.run(debug=True)