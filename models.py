#!/usr/bin/env python3

import pandas as pd
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor


COLUMN_NAMES =["PLAYER", "COLLEGE", "CONF", "GAMES_PLAYED", "GAMES_STARTED", "RECORD", "CMP", "ATT", "YDS", "TD", "INT", "RUSH_ATT", "RUSH_YDS", "RUSH_TD", "COLL_CMP", "COLL_ATT", "COLL_YDS", "COLL_TD", "COLL_INT", "COLL_RUSH_ATT", "COLL_RUSH_YDS", "COLL_RUSH_TDS"]
df = pd.read_csv('data/scraped_data.csv', usecols=COLUMN_NAMES)

REGRESSOR_COLUMN_NAMES = ["GAMES_PLAYED", "GAMES_STARTED", "CMP", "ATT", "YDS", "TD", "INT", "RUSH_ATT", "RUSH_YDS", "RUSH_TD"]
CLASS_LABELS = df['RECORD'].apply(lambda rec: 0 if int(rec.split('-')[0]) < 10 else 1)
REGRESSOR_LABELS = df[REGRESSOR_COLUMN_NAMES]
TRAIN_DATA = df.drop(columns=REGRESSOR_COLUMN_NAMES)
TRAIN_DATA.drop(columns=['PLAYER', 'RECORD'], inplace=True)

#one-hot encoding
# first, combine the 2 dataframes
TEST_COLUMN_NAMES =["PLAYER", "COLLEGE", "CONF", "COLL_CMP", "COLL_ATT", "COLL_YDS", "COLL_TD", "COLL_INT", "COLL_RUSH_ATT", "COLL_RUSH_YDS", "COLL_RUSH_TDS"]
test_df = pd.read_csv('data/test_data.csv', usecols=TEST_COLUMN_NAMES)
combined_df = TRAIN_DATA.append(test_df.drop(columns=["PLAYER"]), ignore_index=True)

# one-hot encoding
one_hot = pd.get_dummies(combined_df)

ENCODED_TRAIN_DATA = one_hot.iloc[:225, :]
ENCODED_TEST_DATA = one_hot.iloc[225:, :]
college_test_df = test_df.merge(ENCODED_TEST_DATA)
college_test_df.drop(columns=['COLLEGE', 'CONF'], inplace=True)

# now I have the following datframes:
#
# TRAIN_DATA: non-encoded training data
# ENCODED_TRAIN_DATA: encoded training data
# test_df: non-encoded test data
# ENCODED_TEST_DATA: encoded test data
# CLASS_LABELS: classifiers for training data
# REGRESSOR_LABELS: regressor results for training data
# college_test_df: encoded test data with player names as column 0
#

def get_college_player_data(playername):
	row = college_test_df.loc[college_test_df['PLAYER'] == playername]
	return row.iloc[:, 1:]

def classify(prediction):
	pred = int(prediction)
	return 'Bust!' if pred == 0 else 'Success!'

def parse_stats(prediction):
	data = {}
	for (i, v) in enumerate(prediction):
		data[REGRESSOR_COLUMN_NAMES[i]] = int(v)
	return json.dumps(data)

class QBModel:

	def __init__(self):
		self.X = ENCODED_TRAIN_DATA
		self.y = CLASS_LABELS
		self.regr_y = REGRESSOR_LABELS

	def dt_classifier_prediction(self, name):
		dt_model = DecisionTreeClassifier(max_depth=8, random_state=42)
		dt_model.fit(self.X, self.y)
		pred = dt_model.predict(get_college_player_data(name))[0]
		return classify(pred)

	def rf_classifier_prediction(self, name):
		rf_model = RandomForestClassifier(n_estimators=1, random_state=42)
		rf_model.fit(self.X, self.y)
		pred = rf_model.predict(get_college_player_data(name))[0]
		return classify(pred)

	def mlp_classifier_prediction(self, name):
		mlp_model = MLPClassifier(random_state=42, hidden_layer_sizes=(8, 8), max_iter=4000)
		mlp_model.fit(self.X, self.y)
		pred = mlp_model.predict(get_college_player_data(name))[0]
		return classify(pred)

	def rf_regressor_prediction(self, name):
		regr = RandomForestRegressor(max_depth=1, random_state=42, n_estimators=100)
		regr.fit(self.X, self.regr_y)
		pred = regr.predict(get_college_player_data(name))[0]
		return parse_stats(pred)
