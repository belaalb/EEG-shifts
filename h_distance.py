import numpy as np
import argparse
import utils
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from matplotlib.ticker import FuncFormatter


parser = argparse.ArgumentParser(description = 'Saving Matlab as tensor in a pickle file')
parser.add_argument('--feature', type=str, default='psd', metavar='Path', help='features group to be considered: all, psd, coh, amod, psdamod, coham...')
parser.add_argument('--enhancement', type=str, default='raw', metavar='Path', help='Enhancement method: raw, wICA')
parser.add_argument('--task', type=str, default='Task', metavar='Path', help='Task: Task, BL1, BL2')
parser.add_argument('--window', type=str, default='w04_o03', metavar='Path', help='Window length and overlap: w03_o04 or w08_o07')
parser.add_argument('--normalization', type=str, default='none', metavar='Path', help='Normalization: none, beta, gamma')
args = parser.parse_args()


'''
.mat files should be in a folder in the following format: ./data/enhancement_task/window_overlap/
Example: ./data/raw_Task/w04_o3/ 
'''

file_path = './data/' + args.enhancement + '_' + args.task + '/' + args.window + '/'
f_matrix, _ = utils.load_mat_struct(file_path, args.feature, args.window, normalization='no')

baseline2_path = './data/' + args.enhancement + '_BL2/' + args.window + '/'
feat_bl2, _ = utils.load_mat_struct_bl(baseline2_path, args.feature, args.window, normalization='no')

baseline1_path = './data/' + args.enhancement + '_BL1/' + args.window + '/'
feat_bl1, _ = utils.load_mat_struct_bl(baseline1_path, args.feature, args.window, normalization='no')

n_runs = 30
normalizations = ['None', 'Whitening', 'Baseline 1', 'Baseline 2']
n_samples = 300
n_subs = 9
np.random.seed(seed=1)

results_dict = {}

results_dict['None'] = []
results_dict['Whitening'] = []
results_dict['Baseline 1'] = []
results_dict['Baseline 2'] = []

for run in range(n_runs):

	random_samples_idx = np.random.randint(f_matrix.shape[-2], size=n_samples)
	feat = f_matrix[:n_subs, 4:, random_samples_idx, :]
	feat = feat.reshape(feat.shape[0], feat.shape[1]*feat.shape[2], feat.shape[3])

	for norm in normalizations:
		print('Starting {} normalization!!'.format(norm))
		h_matrix = np.zeros([n_subs, n_subs])

		for s1 in range(n_subs):
			data_s1 = feat[s1, :, :]
			label_s1 = np.zeros([data_s1.shape[0], 1])

			if norm == 'Whitening':
				scaler = StandardScaler()
				data_s1 = scaler.fit_transform(data_s1)	
			elif (norm == 'Baseline 1') or (norm == 'Baseline 2'):

				if norm == 'Baseline 1':
					bl_data = feat_bl1[s1, 4:, :, :]
				else:
					bl_data = feat_bl2[s1, 4:, :, :]

				bl_data = bl_data.reshape(bl_data.shape[0]*bl_data.shape[1], bl_data.shape[-1])
				scaler = StandardScaler()
				scaler.fit(bl_data)
				data_s1 = scaler.transform(data_s1)

			for s2 in range(n_subs):
				data_s2 = feat[s2, :, :]
				label_s2 = np.ones([data_s2.shape[0], 1])

				if norm == 'Whitening':
					scaler = StandardScaler()
					data_s2 = scaler.fit_transform(data_s2)	
				elif (norm == 'Baseline 1') or (norm == 'Baseline 2'):

					if norm == 'Baseline 1':
						bl_data = feat_bl1[s2, 4:, :, :]
					else:
						bl_data = feat_bl2[s2, 4:, :, :]

					bl_data = bl_data.reshape(bl_data.shape[0]*bl_data.shape[1], bl_data.shape[-1])
					scaler = StandardScaler()
					scaler.fit(bl_data)
					data_s2 = scaler.transform(data_s2)

				x = np.vstack((data_s1, data_s2))
				y = np.vstack((label_s1, label_s2))
				model = RandomForestClassifier(n_estimators = 20)
				acc = cross_val_score(model, x, y.ravel(), cv=5, scoring='accuracy')
				h_dist = max(np.mean(acc), np.mean(1-acc))
				h_matrix[s1, s2] = h_dist 

		fmt = lambda x, pos: '{:.3f}'.format(x)
		disparity_hmap = sns.heatmap(h_matrix, annot=True, fmt='.3f', cmap="RdPu", cbar_kws={'format': FuncFormatter(fmt)})
		plt.savefig(norm+'_distance_hmap_pink.png')
		plt.show()

		cov_shift = utils.covshift_normalized_frob_norm(h_matrix)
		results_dict[norm].append(cov_shift)
		print('Norm of H-distance matrix:', cov_shift)

#pd.DataFrame.from_dict(results_dict).to_csv('./Results/all_normalizations_covariateshift.csv', index=False)


