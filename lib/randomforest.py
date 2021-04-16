#!/usr/bin/python3

print('Loading modules...')

import os, sys, getopt, datetime
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder,FunctionTransformer, PowerTransformer, StandardScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_absolute_error, make_scorer
from pickle import load, dump

import mglearn 
import numpy as np
import pandas as pd
from get_transformer_feature_names import *
from pscore import *

#set the working directory 
os.chdir('/path/to/directory/')

#set seed
np.random.seed(0)

def main():
	if len(sys.argv) < 3 :
		print("Not enough arguments specified\n Usage: randomforest.py <x features path> <y target path> <savepath>")
		sys.exit (1)
	else:
    # print command line arguments
		for arg in sys.argv[0:]:
			print(arg)
	#Load X features data
		X_path = sys.argv[1]
		print('Loading the X features at {}'.format(X_path))
		X_train = pd.read_csv(X_path, index_col = 0)
		X_train = X_train.sort_index(axis = 0)	
	#Load Y target data- this should be just one of the dxa outputs 
		Y_path= sys.argv[2]
		print('Loading Y target at {}'.format(Y_path))
		y_train = pd.read_csv(Y_path, index_col = 0)
		y_train = y_train.sort_index(axis=0)
		y_target = y_train[:-1].columns

	#Load the numeric and categorical feature names 
		num_feat = pd.read_csv("data/numerical_features_ffq.csv", delimiter=',', header=0)
		cat_feat = pd.read_csv("data/categorical_features_ffq.csv", delimiter=',', header=0)
		zero_feat = pd.read_csv("data/ffq_var_with_zeroes.csv", delimiter = ",", header=0)				

	#Define the numeric and categorical features 
		numerical_features = [col for col in X_train.columns if col in num_feat.values]
		categorical_features= [col for col in X_train.columns if col in cat_feat.values]
		numeric_nonzero = [col for col in numerical_features if col not in zero_feat.values]
		numeric_zeroes = [col for col in X_train.columns if col in zero_feat.values]

#Set up the pipeline for numeric variables with no zeroes 
		numeric_transformer = Pipeline(steps = [
			('log',FunctionTransformer(np.log)), #log transform
			('scaler', StandardScaler())
		])

		zero_transformer = Pipeline(steps = [
			('yeo', PowerTransformer(method="yeo-johnson", standardize=True))])
		
		#set up categorical transformer
		X_cat = X_train[categorical_features]
		enc = OneHotEncoder(handle_unknown="error", sparse=False)
		enc.fit(X_cat)
		enc.transform(X_cat)
		cat_levels=enc.categories_

		categorical_transformer = Pipeline(steps = [
    		('onehot', OneHotEncoder(handle_unknown='error',sparse=False, categories=cat_levels))
		])

	#Set up ColumnTransformer
		prep = ColumnTransformer(
			transformers=[
				('num', numeric_transformer, numeric_nonzero),
				('yeo', zero_transformer, numeric_zeroes),
				('cat', categorical_transformer, categorical_features)
			]
		)
    
	#make the model
		model = RandomForestRegressor(random_state = 0,
					oob_score = True,
					warm_start = True,
					 n_jobs = -1)
		
	#setting up pipeline
		pipeline= Pipeline(steps = [(
			'preprocessor', prep),
			('rf', TransformedTargetRegressor(model, func = np.log, inverse_func = np.exp))])	

        #set up the parameter grid
		p_grid = {
			'rf__regressor__n_estimators':np.arange(100,800,50),
			'rf__regressor__max_features':np.arange(0.1, 1, 0.05),
			'rf__regressor__max_samples':np.arange(0.2, 1, 0.1)
			}

		cv = KFold(n_splits=10, shuffle=True, random_state=0)
		refit = 'r2'

		pscore = make_scorer(pcc)
		scoring = {'r2':make_scorer(r2_score), 
		'MAE':make_scorer(mean_absolute_error),
		'PCC':pscore}

	#Set up GridSearchCV
	print('Setting up GridSearchCV...')
	grid_search = GridSearchCV(estimator=pipeline, param_grid=p_grid, cv=cv, 
		refit=refit, n_jobs = -1,
		scoring=scoring,
		return_train_score=True)

	
	#fit the model 
	print('Fitting the model using {}'.format(X_path), 'and {}'.format(Y_path))
	grid_search.fit(X_train, y_train.values.ravel())
	
	#print the results  
	print("\n\nBest params: {}".format(grid_search.best_params_))
	print("Best cross-validation score: {:.3f}".format(grid_search.best_score_))
	print("OOB score: {:.3f}".format(grid_search.best_estimator_.named_steps['rf'].regressor_.oob_score_))
	print("Best estimator:\n{}:".format(
    grid_search.best_estimator_.named_steps['rf']))   

    #save the results
	
	savepath = sys.argv[3]
	
	results = pd.DataFrame(grid_search.cv_results_)
	name = 'cv_results_for_{}'.format(y_target[0])+'.csv'
	path = savepath + name
	results.to_csv(path, index=True)
	
	#save the oob score
	oobscore = pd.Series(grid_search.best_estimator_.named_steps['rf'].regressor_.oob_score_)
	oobname = 'oob_score_for_{}'.format(y_target[0])+'.csv'
	oobpath = savepath + oobname
	oobscore.to_csv(oobpath, index = True)

	#save the model 
	mod_name = 'randomforest_{}'.format(y_target[0])+'.pkl'
	filename = savepath + mod_name
	dump(grid_search, open(filename, 'wb'))
	
	print("\nResults saved to {}".format(path))
	print("\nModel saved to {}".format(filename))

if __name__ == "__main__":
    main()
