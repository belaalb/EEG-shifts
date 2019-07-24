import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn import manifold
import utils
from sklearn import decomposition


parser = argparse.ArgumentParser(description = 'Saving Matlab as tensor in a pickle file')
parser.add_argument('--feature', type=str, default='psd', metavar='Path', help='features group to be considered: all, psd, coh, amod, psdamod, coham...')
parser.add_argument('--enhancement', type=str, default='wICA', metavar='Path', help='Enhancement method: raw, wICA')
parser.add_argument('--normalization', type=str, default='no', metavar='Path', help='Normalization: no, beta, gamma')
parser.add_argument('--task', type=str, default='Task', metavar='Path', help='Task: Task, BL1, BL2')
parser.add_argument('--window', type=str, default='w04_o03', metavar='Path', help='Window length and overlap: w03_o04 or w08_o07')
parser.add_argument('--n-best', type=int, default=20, metavar='N', help='Data splitting fraction')
args = parser.parse_args()


'''
.mat files should be in a folder in the following format: ./data/enhancement_task/window_overlap/
Example: ./data/raw_Task/w04_o3/ 
'''

file_path = './data/' + args.enhancement + '_' + args.task + '/' + args.window + '/'

sess_name = args.task
window = args.window
norm = args.normalization

f_matrix, labels = utils.load_mat_struct(file_path, args.feature, sess_name, window, norm)
n_subs = 5 #feat.shape[0]
feat = f_matrix[26:26+n_subs, 4:, :, :]

random_idx = np.random.randint(feat.shape[-2], size=200)

labels0 = np.zeros(n_subs*feat.shape[2])
labels1 = np.ones(n_subs*feat.shape[2])
labels = np.hstack([labels0, labels1])

# Creating subjects labels
sample_sub = feat.shape[2]
y_subs = []
for sub in range(n_subs):
	label = list(sub*np.ones([sample_sub]))
	y_subs = y_subs + label

y_subs = np.asarray(y_subs)
y_subs = np.hstack([y_subs, y_subs])


feat = np.hstack([feat[:, 0, :, :], feat[:, 1, :, :]])
feat = feat.reshape(feat.shape[0]*feat.shape[1], feat.shape[-1])
#feat = (feat - np.min(feat)) / (np.max(feat)-np.min(feat))


#pca = decomposition.PCA(n_components=2)
#pca.fit(feat)
#feat_tsne = feat #pca.transform(feat)

tsne = manifold.TSNE(n_components=2, init='pca')
feat_tsne = tsne.fit_transform(feat)

#iso = manifold.Isomap(n_neighbors=6, n_components=2)
#iso.fit(feat)
#feat_tsne = iso.transform(feat)

color_high = ['navy', 'purple', 'green', 'magenta', 'crimson']
color_low = ['lightblue', 'lavender', 'lightgreen', 'pink', 'coral']

cm = plt.cm.get_cmap('tab20')

for sub in range(n_subs):
	sub_samples_low = feat_tsne[np.where((y_subs == sub) & (labels == 0)), :].squeeze()
	plt.plot(sub_samples_low[random_idx, 0], sub_samples_low[random_idx, 1], 'o', c = color_low[sub])#cm.colors[2*sub])
	sub_samples_high = feat_tsne[np.where((y_subs == sub) & (labels == 1)), :].squeeze()
	plt.plot(sub_samples_high[random_idx, 0], sub_samples_high[random_idx, 1], 'x', c = color_high[sub])#cm.colors[2*sub+1])

save_name = './Results/testing_'+norm+ '_norm_'

fig_name = save_name+'tsne.png'
plt.savefig(fig_name)
plt.clf()
