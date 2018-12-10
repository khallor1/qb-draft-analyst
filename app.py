#!/usr/bin/env python3

import json
from flask import Flask, jsonify, request, render_template, redirect, abort
from models import QBModel

#load model here
# model = QBModel()

model = QBModel()
print(str(model))

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	if request.method == 'GET':
		return redirect('/')
	else:
		name = request.form['name']
		modelName = request.form['model']
		if (modelName == 'Decision Tree Classifier'):
			return model.dt_classifier_prediction(name)
		elif (modelName == 'Random Forest Classifier'):
			return model.rf_classifier_prediction(name)
		elif (modelName == 'MLP Classifier'):
			return model.mlp_classifier_prediction(name)
		elif (modelName == 'Random Forest Regressor'):
			return model.rf_regressor_prediction(name)
		else:
			return abort(404)

if __name__ == '__main__':
	app.run(port=5000, debug=True)
