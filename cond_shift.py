import numpy as np
import argparse
import utils
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


from sklearn.ensemble import RandomForestClassifier

parser = argparse.ArgumentParser(description = 'Saving Matlab as tensor in a pickle file')
parser.add_argument('--feature', type=str, default='psd', metavar='Path', help='features group to be considered: all, psd, coh, amod, psdamod, coham...')
parser.add_argument('--enhancement', type=str, default='raw', metavar='Path', help='Enhancement method: raw, wICA')
parser.add_argument('--window', type=str, default='w04_o03', metavar='Path', help='Window length and overlap: w03_o04 or w08_o07')
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
n_samples = 300
n_subs = 9 
np.random.seed(seed=10)

n_conditions = 2

normalizations = ['None', 'Whitening']#, 'Baseline 1', 'Baseline 2']

results_dict = {}

results_dict['None'] = []
results_dict['Whitening'] = []
results_dict['Baseline 1'] = []
results_dict['Baseline 2'] = []

matrix_dict = {}
matrix_dict['None'] = []
matrix_dict['Whitening'] = []
#matrix_dict['Baseline 1'] = []
#matrix_dict['Baseline 2'] = []


for run in range(n_runs):

	random_samples_idx = np.random.choices(range(f_matrix.shape[-2]), size=n_samples, replace=False)
	valid_samples_idx = [] 

	for x in range(f_matrix.shape[-2]):	
		if x not in random_samples_idx:
			valid_samples_idx.append(x)

	feat_valid = f_matrix[:n_subs, 4:, valid_samples_idx, :]
	feat = f_matrix[:n_subs, 4:, random_samples_idx, :]
	
	for norm in normalizations:
		print('Starting {} normalization!!'.format(norm))

		labels0 = np.zeros(n_samples)
		labels1 = np.ones(n_samples)
		labels = np.hstack([labels0, labels1])

		disparity_matrix = np.zeros([n_subs, n_subs])

		for s2 in range(n_subs):
			model = KNeighborsClassifier(n_neighbors=5)
			#model = RandomForestClassifier(n_estimators = 20)
			sub_feat = np.vstack([feat[s2, 0, :, :], feat[s2, 1, :, :]])

			if norm == 'Whitening':
				scaler = StandardScaler()
				sub_feat = scaler.fit_transform(sub_feat)	
			elif (norm == 'Baseline 1') or (norm == 'Baseline 2'):

				if norm == 'Baseline 1':
					bl_data = feat_bl1[s2, 4:, :, :]
				else:
					bl_data = feat_bl2[s2, 4:, :, :]

		
				bl_data = bl_data.reshape(bl_data.shape[0]*bl_data.shape[1], bl_data.shape[-1])
				scaler = StandardScaler()
				scaler.fit(bl_data)
				sub_feat = scaler.transform(sub_feat)

			model.fit(sub_feat, labels)	

			for s1 in range(n_subs):
				equal_label = 0
				diff_label = 0
				if s2 == s1:
					sub1_feat = feat_valid[s1, :, :, :].reshape(feat_valid.shape[1]*feat_valid.shape[2], feat_valid.shape[-1])
					sub_data = feat_valid
				else:
					sub1_feat = feat[s1, :, :, :].reshape(feat.shape[1]*feat.shape[2], feat.shape[-1])
					sub_data = feat

				if norm == 'Whitening':
					scaler_s1 = StandardScaler()
					scaler_s1 = scaler.fit(sub1_feat)	
				elif (norm == 'Baseline 1') or (norm == 'Baseline 2'):
					if norm == 'Baseline 1':
						bl_data = feat_bl1[s1, 4:, :, :]
					else:
						bl_data = feat_bl2[s1, 4:, :, :]

					bl_data = bl_data.reshape(bl_data.shape[0]*bl_data.shape[1], bl_data.shape[-1])
					scaler_s1 = StandardScaler()
					scaler_s1.fit(bl_data)

				for cond in range(n_conditions):
					for sample in range(n_samples):
						data = sub_data[s1, cond, sample, :].reshape(1, -1)

						if norm != 'None':
							data = scaler_s1.transform(data)
						pred = model.predict(data)
						
						if pred == cond:
							equal_label += 1
						else:
							diff_label += 1

				disparity_matrix[s1, s2] = diff_label/(2*n_samples)

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

df_subs = pd.concat([matrix_dict['None'], matrix_dict['Whitening']])
df_subs.to_csv('./Results/average_disparity_matrix.csv', index=False)

