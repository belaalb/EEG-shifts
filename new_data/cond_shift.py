import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from load_data_utils import build_feat_matrix, load_subject_session
import sys
sys.path.append('../')
import utils


n_runs = 30
n_valid_samples = 30
n_subs = 9 
np.random.seed(seed=10)

normalizations = ['None', 'Whitening', 'Baseline 1', 'Baseline 2']

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
			#model = RandomForestClassifier(n_estimators = 100)
			sub_feat = feat_train[s2]

			if norm == 'Whitening':
				scaler = StandardScaler()
				sub_feat = scaler.fit_transform(sub_feat)	
			elif (norm == 'Baseline 1') or (norm == 'Baseline 2'):
				if norm == 'Baseline 1':
					bl_data, _ = load_subject_session(s2+1, 'baseline1')
				else:
					bl_data, _ = load_subject_session(s2+1, 'baseline2')
		
				scaler = StandardScaler()
				scaler.fit(bl_data)
				sub_feat = scaler.transform(sub_feat)

			model.fit(sub_feat, labels_train[s2])	

			for s1 in range(n_subs):
				equal_label = 0
				diff_label = 0
				if s2 == s1:
					sub1_feat = feat_valid[s1]
					sub1_label = labels_valid[s1]
				else:
					sub1_feat = feat_train[s1]
					sub1_label = labels_train[s1]
					
				if norm == 'Whitening':
					scaler_s1 = StandardScaler()
					scaler_s1 = scaler.fit(sub1_feat)
						
				elif (norm == 'Baseline 1') or (norm == 'Baseline 2'):
					if norm == 'Baseline 1':
						bl_data, _ = load_subject_session(s1+1, 'baseline1')
					else:
						bl_data, _ = load_subject_session(s1+1, 'baseline2')

					scaler_s1 = StandardScaler()
					scaler_s1.fit(bl_data)

				for sample in range(sub1_feat.shape[0]):
					data = sub1_feat[sample, :].reshape(1, -1)
					if norm != 'None':
						data = scaler_s1.transform(data)
					pred = model.predict(data)
					
					if pred == sub1_label[sample]:
						equal_label += 1
					else:
						diff_label += 1

				disparity_matrix[s1, s2] = diff_label/(sub1_feat.shape[0])

		for s1 in range(n_subs):
			for s2 in range(n_subs):
				disparity_matrix[s1, s2] = min(min(disparity_matrix[s1, s2], 1-disparity_matrix[s1, s2]), min(disparity_matrix[s2, s1], 1-disparity_matrix[s2, s1]))
				disparity_matrix[s2, s1] = disparity_matrix[s1, s2] 
		
		matrix_dict[norm].append(disparity_matrix)

		cond_shift = utils.disparity_normalized_frob_norm(disparity_matrix)
		results_dict[norm].append(cond_shift)
		print('Norm of disparity matrix:', cond_shift)

		if len(matrix_dict[norm]) <= 1:
			fmt = lambda x, pos: '{:.3f}'.format(x)
			disparity_hmap = sns.heatmap(disparity_matrix, annot=True, fmt='.3f', cmap="RdPu", cbar_kws={'format': FuncFormatter(fmt)})
			plt.savefig(norm+'_disparity_hmap_pink.png')
			plt.show()

pd.DataFrame.from_dict(results_dict).to_csv('./Results/all_normalizations_condshift.csv', index=False)

# Saving average disparity matrices 
for key in matrix_dict.keys():
	values = np.mean(matrix_dict[key], axis=0)
	matrix_dict[key] = pd.DataFrame(values, columns=['0', '1', '2', '3', '4', '5', '6', '7', '8']).assign(Normalization=key)

df_subs = pd.concat([matrix_dict['None'], matrix_dict['Whitening'], matrix_dict['Baseline 1'], matrix_dict['Baseline 2']])
df_subs.to_csv('./Results/average_disparity_matrix.csv', index=False)

