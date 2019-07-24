import numpy as np
import argparse
import utils
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description = 'Saving Matlab as tensor in a pickle file')
parser.add_argument('--feature', type=str, default='psd', metavar='Path', help='features group to be considered: all, psd, coh, amod, psdamod, coham...')
parser.add_argument('--enhancement', type=str, default='raw', metavar='Path', help='Enhancement method: raw, wICA')
parser.add_argument('--task', type=str, default='Task', metavar='Path', help='Task: Task, BL1, BL2')
parser.add_argument('--window', type=str, default='w04_o03', metavar='Path', help='Window length and overlap: w03_o04 or w08_o07')
parser.add_argument('--normalization', type=str, default='no', metavar='Path', help='Normalization: no, beta, gamma')
args = parser.parse_args()


'''
.mat files should be in a folder in the following format: ./data/enhancement_task/window_overlap/
Example: ./data/raw_Task/w04_o3/ 
'''

file_path = './data/' + args.enhancement + '_' + args.task + '/' + args.window + '/'

sess_name = args.task
window = args.window
normalization = args.normalization

f_matrix, labels = utils.load_mat_struct(file_path, args.feature, sess_name, window, normalization)

feat = f_matrix[:, 4:, :, :]
n_subs = 10 #feat.shape[0]
n_conditions = 2
n_samples = feat.shape[2]

labels0 = np.zeros(n_samples)
labels1 = np.ones(n_samples)
labels = np.hstack([labels0, labels1])

similarity_matrix = np.zeros([n_subs, n_subs])
disparity_matrix = np.zeros([n_subs, n_subs])

for s2 in range(n_subs):
	neigh = KNeighborsClassifier(n_neighbors=1)
	sub_feat = feat[s2, :, :, :].reshape(feat.shape[1]*feat.shape[2], feat.shape[-1]) 	
	neigh.fit(sub_feat, labels)	

	for s1 in range(n_subs):
		equal_label = 0
		diff_label = 0
		for cond in range(n_conditions):
			for sample in range(n_samples):
				data = feat[s1, cond, sample, :].reshape(1, -1)
				pred = neigh.predict(data)
				if pred == cond:
					equal_label += 1
				else:
					diff_label += 1
	
		similarity_matrix[s1, s2] = equal_label/(2*n_samples)
		disparity_matrix[s1, s2] = diff_label/(2*n_samples)


for s1 in range(n_subs):
	for s2 in range(n_subs):
		similarity_matrix[s1, s2] = min(similarity_matrix[s1, s2], similarity_matrix[s2, s1])
		similarity_matrix[s2, s1] = similarity_matrix[s1, s2]

		disparity_matrix[s1, s2] = max(disparity_matrix[s1, s2], disparity_matrix[s2, s1])
		disparity_matrix[s2, s1] = disparity_matrix[s1, s2] 
		

print('Norm of disparity matrix:', utils.disparity_normalized_frob_norm(disparity_matrix))
print('Norm of similarity matrix:', utils.similarity_normalized_frob_norm(similarity_matrix))

save_name = './Results/'+normalization + '_norm_'

similarity_hmap = sns.heatmap(similarity_matrix, yticklabels=False, cmap="YlGnBu")
plt.savefig(save_name+'similarity_hmap.png')
plt.show()

disparity_hmap = sns.heatmap(disparity_matrix, yticklabels=False, cmap="YlGnBu")
plt.savefig(save_name+'disparity_hmap.png')
plt.show()

pd.DataFrame(similarity_matrix).to_csv(save_name+'similarity_matrix.csv', float_format='%.3f')
pd.DataFrame(disparity_matrix).to_csv(save_name+'disparity_matrix.csv', float_format='%.3f')

