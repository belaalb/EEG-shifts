import numpy as np
import argparse
import utils
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



parser = argparse.ArgumentParser(description = 'Saving Matlab as tensor in a pickle file')
parser.add_argument('--feature', type=str, default='psd', metavar='Path', help='features group to be considered: all, psd, coh, amod, psdamod, coham...')
parser.add_argument('--enhancement', type=str, default='wICA', metavar='Path', help='Enhancement method: raw, wICA')
parser.add_argument('--window', type=str, default='w04_o03', metavar='Path', help='Window length and overlap: w03_o04 or w08_o07')
parser.add_argument('--normalization', type=str, default='none', metavar='Path', help='Normalization: no, white, bl1, bl2')
args = parser.parse_args()


'''
.mat files should be in a folder in the following format: ./data/enhancement_task/window_overlap/
Example: ./data/raw_Task/w04_o3/ 
'''

file_path = './data/' + args.enhancement + '_Task/' + args.window + '/'
f_matrix, _ = utils.load_mat_struct(file_path, args.feature, args.window, normalization='no')

baseline2_path = './data/' + args.enhancement + '_BL2/' + args.window + '/'
feat_bl2, _ = utils.load_mat_struct_bl(baseline2_path, args.feature, args.window, normalization='no')

baseline1_path = './data/' + args.enhancement + '_BL1/' + args.window + '/'
feat_bl1, _ = utils.load_mat_struct_bl(baseline1_path, args.feature, args.window, normalization='no')

n_runs = 30
n_subs = 9
n_samples = 300

np.random.seed(seed=10)

normalizations = ['None', 'Whitening', 'Baseline 1', 'Baseline 2']

results_dict_test = {}

results_dict_test['None'] = []
results_dict_test['Whitening'] = []
results_dict_test['Baseline 1'] = []
results_dict_test['Baseline 2'] = []

results_dict_train = {}

results_dict_train['None'] = []
results_dict_train['Whitening'] = []
results_dict_train['Baseline 1'] = []
results_dict_train['Baseline 2'] = []

for run in range(n_runs):
	random_samples_idx = np.random.randint(f_matrix.shape[-2], size=n_samples)
	feat = f_matrix[:n_subs, 4:, random_samples_idx, :]

	for norm in normalizations:
		print(norm)
		# Creating subjects labels
		sample_sub = feat.shape[2]
		y_subs = []
		for sub in range(n_subs):
			label = list(sub*np.ones([sample_sub]))
			y_subs = y_subs + label

		y_subs = np.asarray(y_subs)
		y_subs = np.hstack([y_subs, y_subs])

		feat_norm = np.copy(feat)

		for sub in range(n_subs):				
			sub_data = feat[sub, :, :, :]
			sub_data = sub_data.reshape(sub_data.shape[0]*sub_data.shape[1], sub_data.shape[-1])
			if norm == 'Whitening':
				scaler = StandardScaler()
				scaler = scaler.fit(sub_data)
				feat_norm[sub,0,:,:] = scaler.transform(feat[sub,0,:,:])
				feat_norm[sub,1,:,:] = scaler.transform(feat[sub,1,:,:])
			elif (norm == 'Baseline 1') or (norm == 'Baseline 2'): 
				if norm == 'Baseline 1':
					bl_data = feat_bl1[sub, 4:, :, :]
				elif norm == 'Baseline 2':
					bl_data = feat_bl2[sub, 4:, :, :]

				bl_data = bl_data.reshape(bl_data.shape[0]*bl_data.shape[1], bl_data.shape[-1])
				scaler = StandardScaler()
				scaler.fit(bl_data)
				feat_norm[sub,0,:,:] = scaler.transform(feat[sub,0,:,:])
				feat_norm[sub,1,:,:] = scaler.transform(feat[sub,1,:,:])

		feat_low = feat_norm[:, 0, :, :]
		feat_high = feat_norm[:, 1, :, :]

		feat_low = feat_low.reshape(feat_low.shape[0]*feat_low.shape[1], feat_low.shape[-1])
		feat_high = feat_high.reshape(feat_high.shape[0]*feat_high.shape[1], feat_high.shape[-1])
		label_low = np.zeros([feat_low.shape[0], 1])
		label_high = np.ones([feat_high.shape[0], 1])

		x = np.vstack((feat_low, feat_high))
		y = np.vstack((label_low, label_high))

		logo = LeaveOneGroupOut()
		logo.get_n_splits(x, y, y_subs)
		logo.get_n_splits(groups=y_subs)

		acc_list = []

		for train_index, test_index in logo.split(x, y, y_subs):

			model = RandomForestClassifier(n_estimators = 20)
			#acc = cross_val_score(model, x[train_index, :], y[train_index, :].ravel(), cv=5, scoring='accuracy')
			#acc_train = max(np.mean(acc), 1-np.mean(acc))
			#results_dict_train[norm].append(np.asarray(acc_train))			

			x_train, x_source_test, y_train, y_source_test = train_test_split(x[train_index, :], y[train_index, :].ravel(), test_size=0.33, stratify=y_subs[train_index])
			
			acc_test, acc_train = utils.calculate_metrics(model, x_train, y_train, x[test_index, :], y[test_index, :].ravel(), x_source_test, y_source_test)
			acc_test = max(acc_test, 1-acc_test)
			results_dict_test[norm].append(np.asarray(acc_test))
			acc_train = max(acc_train, 1-acc_train)
			results_dict_train[norm].append(np.asarray(acc_train))

pd.DataFrame.from_dict(results_dict_train).to_csv('./Results/all_normalizations_classification_train.csv', index=False)
pd.DataFrame.from_dict(results_dict_test).to_csv('./Results/all_normalizations_classification_test.csv', index=False)


