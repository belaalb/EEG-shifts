import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import accuracy_score

from load_data_utils import build_feat_matrix, load_subject_session
import sys
sys.path.append('../')
import utils


n_runs = 1
n_valid_samples = 30
n_subs = 5 
np.random.seed(seed=10)

normalizations = ['None']#, 'Whitening', 'Baseline 1', 'Baseline 2']

results_dict = {}

results_dict['None'] = []
results_dict['Whitening'] = []
results_dict['Baseline 1'] = []
results_dict['Baseline 2'] = []

matrix_dict = {}
matrix_dict['None'] = []
matrix_dict['Whitening'] = []
matrix_dict['Baseline 1'] = []
matrix_dict['Baseline 2'] = []


for run in range(n_runs):

	feat_train, labels_train, feat_valid, labels_valid = build_feat_matrix(range(1, n_subs+1), 'task', n_valid_samples)
		
	for norm in normalizations:
		print('Starting {} normalization!!'.format(norm))

		disparity_matrix = np.zeros([n_subs, n_subs])

		for s2 in range(n_subs):
			model = KNeighborsClassifier(n_neighbors=3)
			#model = RandomForestClassifier(n_estimators = 50)
			
			sub_feat = feat_train[s2]
			sub_feat_valid = feat_valid[s2]

			if norm == 'Whitening':
				scaler = StandardScaler()
				sub_feat = scaler.fit_transform(sub_feat)
				
				scaler_valid = StandardScaler()
				sub_feat_valid = scaler_valid.fit_transform(sub_feat_valid)
				 	
			elif (norm == 'Baseline 1') or (norm == 'Baseline 2'):
				if norm == 'Baseline 1':
					bl_data, _ = load_subject_session(s2+1, 'baseline1')
				else:
					bl_data, _ = load_subject_session(s2+1, 'baseline2')
		
				scaler = StandardScaler()
				scaler.fit(bl_data)
				sub_feat = scaler.transform(sub_feat)
				sub_feat_valid = scaler.transform(sub_feat_valid)

			model.fit(sub_feat, labels_train[s2])
			pred_train = model.predict(sub_feat)
			pred_valid = model.predict(sub_feat_valid)	

			acc_train = accuracy_score(labels_train[s2], pred_train)
			acc_valid = accuracy_score(labels_valid[s2], pred_valid)

			print('acc train {}'.format(acc_train))
			print('acc valid {}'.format(acc_valid))

