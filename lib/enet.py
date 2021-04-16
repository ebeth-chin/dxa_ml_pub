#!/usr/bin/python3

print('Loading modules...')

import os, sys, getopt, datetime
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder,FunctionTransformer, PowerTransformer, StandardScaler 
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_absolute_error, make_scorer
from pickle import load, dump

import numpy as np
import pandas as pd
import scipy as sp
from get_transformer_feature_names import *
from pscore import *

#set the working directory 
os.chdir('/path/to/directory/')

#set seed
np.random.seed(0)

def main():
	if len(sys.argv) < 3 :
		print("Not enough arguments specified\n Usage: enet.py <x features path> <y target path> <outdir>")
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
		numeric_nonzero = [col for col in numerical_features if col not in zero_feat.values]
		numeric_zeroes = [col for col in X_train.columns if col in zero_feat.values]
		categorical_features= [col for col in X_train.columns if col in cat_feat.values]

		print('Setting up ColumnTransformer...')
		numeric_transformer = Pipeline(steps = [
			('log',FunctionTransformer(np.log)), 
			('scaler', StandardScaler())
		])

	#set up pipeline for numeric variables with zeroes
		zero_transformer = Pipeline(steps = [
			('yeo', PowerTransformer(method="yeo-johnson", standardize=True))])
		
	#Set up the categorical pipeline
		#define the unique levels of each category
		X_cat = X_train[categorical_features]
		enc = OneHotEncoder(handle_unknown="error", sparse=False)
		enc.fit(X_cat)
		enc.transform(X_cat)
		cat_levels=enc.categories_
		#define the categorical transformer
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
		
		model = TransformedTargetRegressor(ElasticNet(random_state = 0, normalize=False), func = np.log, inverse_func = np.exp)	

	#Set up the pipeline
		print('Setting up pipeline...')
		pipeline= Pipeline(steps = [(
			'preprocessor', prep),
			('elasticnet',model)])

	#Set up the param grid and CV 
		param_grid = {'elasticnet__regressor__alpha':np.logspace(-4, 0,50),
                        'elasticnet__regressor__l1_ratio':[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]}

	#define inner and outer cv
	inner_cv = KFold(n_splits=10, shuffle=True, random_state=0)
	outer_cv = KFold(n_splits=10, shuffle=True, random_state=0) 
	
	refit = 'r2'
	pscore = make_scorer(pcc)
	scoring = {'r2':make_scorer(r2_score), 
	        'MAE':make_scorer(mean_absolute_error),
			'pearson':pscore}
	
	#create output sinks
	outer_loop_r2 = []
	outer_loop_pcc = []
	outer_loop_mae = []

	inner_loop_won_params = []
	inner_loop_accuracy_scores = []
	inner_loop_coefs = []
	inner_loop_best_cv_results = []

# Looping through the outer loop, feeding each training set into a grid_search as the inner loop
	for train_index,test_index in outer_cv.split(X_train,y_train):
	
		grid_search = GridSearchCV(estimator=pipeline,param_grid=param_grid,cv=inner_cv, scoring = scoring, refit = "r2", n_jobs = -1)
	
		# inner loop
		grid_search.fit(X_train.iloc[train_index],y_train.iloc[train_index])
		inner_results = pd.DataFrame(grid_search.cv_results_)
		inner_best_scores = inner_results[inner_results['rank_test_r2']==1]
	
		# The best hyper parameters from grid_search is now being tested on the unseen outer loop test data.
		pred = grid_search.predict(X_train.iloc[test_index])
	
		# Appending the "winning" hyper parameters and their associated accuracy score
		outer_loop_r2.append(r2_score(y_train.iloc[test_index],pred))
		outer_loop_mae.append(mean_absolute_error(y_train.iloc[test_index],pred))
		outer_loop_pcc.append(sp.stats.pearsonr(y_train.iloc[test_index],pred)[0])

		inner_loop_won_params.append(grid_search.best_params_)
		inner_loop_best_cv_results.append(inner_best_scores)
		inner_loop_coefs.append(grid_search.best_estimator_.named_steps['elasticnet'].regressor_.coef_)
		inner_loop_accuracy_scores.append(grid_search.best_score_)
	

	for i in zip(inner_loop_won_params,outer_loop_r2,inner_loop_accuracy_scores):
		print(i)

	print('Mean of outer loop accuracy score:',np.mean(outer_loop_r2))

	#save the results
	
	cv_savepath = sys.argv[3]
	
	#save outer loop scores 
	outer_results = pd.DataFrame()
	outer_results['r2'] = outer_loop_r2
	outer_results['mae']= outer_loop_mae
	outer_results['pcc']= outer_loop_pcc
	outer_results['pcc'] = outer_results['pcc'].str.get(0)
	outer_name = 'outer_loop_results_for_{}'.format(y_target[0]) + '.csv'
	outer_path = cv_savepath + outer_name
	outer_results.to_csv(outer_path, index = True)
	
	#save the inner loop results 
	inner_results = pd.concat(inner_loop_best_cv_results)
	inner_name = 'inner_loop_results_for_{}'.format(y_target[0]) + '.csv'
	inner_path = cv_savepath + inner_name
	inner_results.to_csv(inner_path, index = True)
	
	#get the feature names
	prep.fit(X_train)
	feature_names = get_transformer_feature_names(prep)
	#save the inner loop coefs 
	inner_feat_df = pd.DataFrame(inner_loop_coefs).T
	inner_feat_df['Feature'] = feature_names
	inner_feat_df = inner_feat_df.set_index(['Feature'])
	inner_coef_name = 'inner_loop_coefs_for_{}'.format(y_target[0]) + '.csv'
	inner_coef_path = cv_savepath + inner_coef_name
	inner_feat_df.to_csv(inner_coef_path, index = True)

	#save the model 
	mod_name = 'enet_{}'.format(y_target[0])+'.pkl'
	filename = cv_savepath + mod_name
	dump(grid_search.best_estimator_, open(filename, 'wb'))
 
	print("\nResults saved to {}".format(cv_savepath))
	print("\nModel saved to {}".format(filename))
		
if __name__ == "__main__":
	main()
