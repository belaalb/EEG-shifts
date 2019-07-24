import numpy as np
import argparse
import utils
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


parser = argparse.ArgumentParser(description = 'Saving Matlab as tensor in a pickle file')
parser.add_argument('--feature', type=str, default='psd', metavar='Path', help='features group to be considered: all, psd, coh, amod, psdamod, coham...')
parser.add_argument('--enhancement', type=str, default='wICA', metavar='Path', help='Enhancement method: raw, wICA')
parser.add_argument('--task', type=str, default='Task', metavar='Path', help='Task: Task, BL1, BL2')
parser.add_argument('--window', type=str, default='w04_o03', metavar='Path', help='Window length and overlap: w03_o04 or w08_o07')
parser.add_argument('--normalization', type=str, default='none', metavar='Path', help='Normalization: none, beta, gamma')
args = parser.parse_args()


'''
.mat files should be in a folder in the following format: ./data/enhancement_task/window_overlap/
Example: ./data/raw_Task/w04_o3/ 
'''

file_path = './data/' + args.enhancement + '_' + args.task + '/' + args.window + '/'

sess_name = args.task
window = args.window
n_runs = 1

normalizations = ['no', 'beta', 'gamma']

results_dict = {}

for norm in normalizations:
	print('Starting {} normalization!!'.format(norm))

	f_matrix, _ = utils.load_mat_struct(file_path, args.feature, sess_name, window, norm)
	results_dict[norm] = []	

	for run in range(n_runs):

		n_samples = 600
		random_samples_idx = np.random.randint(f_matrix.shape[-2], size=n_samples)

		n_subs = 10
		feat = f_matrix[:n_subs, 4:, random_samples_idx, :]

		feat = feat.reshape(feat.shape[0], feat.shape[1]*feat.shape[2], feat.shape[3])

		for sub in range(n_subs):
			feat[sub,:,:] = utils.remove_outliers(feat[sub,:,:])

		# Scaling to 0-1
		feat = (feat - np.min(feat)) / (np.max(feat)-np.min(feat))

		frechet_matrix = np.zeros([n_subs, n_subs])

		for s2 in range(n_subs):
			for s1 in range(n_subs):
				data_s1 = feat[s1, :, :]
				data_s2 = feat[s2, :, :]

				frechet_dist = utils.compute_fd(data_s1, data_s2)
				frechet_matrix[s1, s2] = frechet_dist 

		#print(frechet_matrix)
		cov_shift = np.linalg.norm(frechet_matrix)
		results_dict[norm].append(cov_shift)
		print('Norm of Frechet distance matrix:', cov_shift)


pd.DataFrame.from_dict(results_dict).to_csv('./Results/all_normalizations_covariateshift.csv', index=False)



