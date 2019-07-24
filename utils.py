import scipy.io as sio
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
from scipy import linalg as sla
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def calculate_metrics(model, x_train, y_train, x_test, y_test, x_source_test, y_source_test):

	#pca = PCA(n_components=8)
	#x_train = pca.fit_transform(x_train)
	#x_test = pca.transform(x_test)


	model.fit(x_train, y_train)
	y_pred, y_score = model.predict(x_test), model.predict_proba(x_test)[:, 1]
	acc_test = metrics.accuracy_score(y_test, y_pred)
	
	y_pred_train, y_score_train = model.predict(x_source_test), model.predict_proba(x_source_test)[:, 1]
	acc_train = metrics.accuracy_score(y_source_test, y_pred_train)	

	return acc_test, acc_train

def remove_outliers(feat_matrix):
	outliers_fraction = 0.01
	algorithm = LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction)
	algorithm.fit(feat_matrix)
	outliers = algorithm.fit_predict(feat_matrix)
	print(np.sum(outliers==-1))
	clean_matrix = np.copy(feat_matrix)
	clean_matrix[np.where(outliers == -1)] = 0

	return clean_matrix	

def compute_fd(x1, x2):
	m1 = x1.mean(0)
	C1 = np.cov(x1, rowvar=False)

	#print(C1.shape[0], np.linalg.matrix_rank(C1))

	m2 = x2.mean(0)
	C2 = np.cov(x2, rowvar=False)

	#print(C2.shape[0], np.linalg.matrix_rank(C2))

	fd = ((m1 - m2) ** 2).sum() + np.matrix.trace(C1 + C2 - 2 * sla.sqrtm(np.matmul(C1, C2)))

	return fd

def covshift_normalized_frob_norm(matrix):
	frob = np.linalg.norm(matrix)
	N = float(matrix.shape[0])
	normalized_frob = (frob - 0.5*N) / (np.sqrt(N**2-0.75*N) - 0.5*N)	

	return normalized_frob

def similarity_normalized_frob_norm(matrix):
	frob = np.linalg.norm(matrix)
	N = float(matrix.shape[0])
	normalized_frob = (frob - np.sqrt(N)) / (N - np.sqrt(N))

	return normalized_frob

def disparity_normalized_frob_norm(matrix):
	frob = np.linalg.norm(matrix)
	N = float(matrix.shape[0])
	normalized_frob = 2*frob / N

	return normalized_frob

def load_mat_struct(matfile_path, feature, window, normalization, check_abn = False):
	'''
	This function loads .mat structures (amod.mat, psd.mat, coh.mat, and coham.mat)
	as a tensor with the following shape:
	[n_subjects, n_sessions, n_samples, n_features]
	n_samples is the minimum number of samples in all sessions for all subjects.
	'''

	feat_path = matfile_path + feature + '.mat'
	data_dict = sio.loadmat(feat_path)
	labels = data_dict['class_vector'].squeeze()
	labels_list = list(labels)

	if (feature == 'amod') | (feature == 'psd'):
		if normalization == 'no':
			idx = [0,1,2,3,10,11,12,13,20,21,22,23,30,31,32,33] # Frontal PSD features indexes
			feat_matrix = data_dict['ft_matrix'].squeeze()
			feat_matrix = feat_matrix
			names = data_dict['ft_names'].squeeze()

		elif normalization == 'beta':
			idx = [0,1,2,3,7,8,9,10,14,15,16,17,21,22,23,24]
			feat_matrix = data_dict['ft_matrix_relative_beta'].squeeze()
			names = data_dict['ft_names_beta'].squeeze()

		elif normalization == 'gamma':
			idx = [0,1,2,3,9,10,11,12,18,19,20,21,27,28,29,30]
			feat_matrix = data_dict['ft_matrix_relative_gamma'].squeeze()
			names = data_dict['ft_names_gamma'].squeeze()
		else:
			print('{} is not a normalization!!'.format(normalization))		

	elif (feature == 'coham') | (feature == 'coh'):
		feat_matrix = data_dict['ft_matrix_gamma'].squeeze()


	names_list = list(names)
	name = names_list[0]
	list_ = list(feat_matrix)

	min_samples = min([x.shape[0] for x in list_])
	list_ = [x[:min_samples, idx] for x in list_]
	feat_matrix = np.asarray(list_)

	labels_list = [x[:min_samples, :] for x in labels_list]
	labels = np.asarray(labels_list).squeeze()

	if check_abn:
		check_abnormalities(feat_matrix, names_list)
	
	feat_matrix = feat_matrix.reshape(feat_matrix.shape[0]//6, 6, feat_matrix.shape[1], feat_matrix.shape[2])

	labels = labels.reshape(labels.shape[0]//6, 6*labels.shape[1])

	return feat_matrix, labels

def load_mat_struct_bl(matfile_path, feature, window, normalization, check_abn = False):
	'''
	This function loads .mat structures (amod.mat, psd.mat, coh.mat, and coham.mat)
	as a tensor with the following shape:
	[n_subjects, n_sessions, n_samples, n_features]
	n_samples is the minimum number of samples in all sessions for all subjects.
	'''

	feat_path = matfile_path + feature + '.mat'
	data_dict = sio.loadmat(feat_path)
	labels = data_dict['class_vector'].squeeze()
	labels_list = list(labels)

	if (feature == 'amod') | (feature == 'psd'):
		if normalization == 'no':
			idx = [0,1,2,3,10,11,12,13,20,21,22,23,30,31,32,33]	# Frontal PSD features indexes
			feat_matrix = data_dict['ft_matrix'].squeeze()
			feat_matrix = feat_matrix
			names = data_dict['ft_names'].squeeze()

		elif normalization == 'beta':
			idx = [0,1,2,3,7,8,9,10,14,15,16,17,21,22,23,24]
			feat_matrix = data_dict['ft_matrix_relative_beta'].squeeze()
			names = data_dict['ft_names_beta'].squeeze()

		elif normalization == 'gamma':
			idx = [0,1,2,3,9,10,11,12,18,19,20,21,27,28,29,30]
			feat_matrix = data_dict['ft_matrix_relative_gamma'].squeeze()
			names = data_dict['ft_names_gamma'].squeeze()
		else:
			print('{} is not a normalization!!'.format(normalization))		

	elif (feature == 'coham') | (feature == 'coh'):
		feat_matrix = data_dict['ft_matrix_gamma'].squeeze()


	names_list = list(names)
	name = names_list[0]
	list_ = list(feat_matrix)

	min_samples = min([x.shape[0] for x in list_])
	list_ = [x[:min_samples, idx] for x in list_]

	subs_to_include = [0, 3, 4, 5, 6, 7, 9, 11, 12]

	all_subs_to_include = []

	for k in subs_to_include:	
		all_subs_to_include.append(6*k)
		all_subs_to_include.append(6*k+1)
		all_subs_to_include.append(6*k+2)
		all_subs_to_include.append(6*k+3)
		all_subs_to_include.append(6*k+4)
		all_subs_to_include.append(6*k+5)		

	new_list = []

	for j in all_subs_to_include:
		new_list.append(list_[j])


	feat_matrix = np.asarray(new_list)

	labels_list = [x[:min_samples, :] for x in labels_list]
	labels = np.asarray(labels_list).squeeze()

	if check_abn:
		check_abnormalities(feat_matrix, names_list)
	
	feat_matrix = feat_matrix.reshape(feat_matrix.shape[0]//6, 6, feat_matrix.shape[1], feat_matrix.shape[2])

	#labels = labels.reshape(labels.shape[0]//6, 6*labels.shape[1])

	return feat_matrix, labels

def check_abnormalities(feat_matrix_, names_):
	# Checking for abnormalites in features

	for feat in range(feat_matrix_.shape[2]):

		prob_subs1 = []
		prob_subs2 = []
		prob_subs3 = []	

		for sub in range(feat_matrix_.shape[0]):

			feat_slice = feat_matrix_[sub, :, feat]	
			feat_max = np.max(feat_slice)
			feat_min = np.min(feat_slice)
			diff = np.round(feat_max - feat_min)
		
			if (diff < 0.001):
				prob_subs1.append(sub//6 + 1)

			std_ = np.std(feat_slice)
			if (std_ < 0.001):
				prob_subs2.append(sub//6 + 1)

			
			if (np.any(np.isnan(feat_slice))):
				prob_subs3.append(sub//6 + 1)

		#if (len(prob_subs1) > 0):

		#	print(names_[feat])
		#	print('Subjects with abnormal features', set(prob_subs1))

		#if ((len(prob_subs2) > 0)):

		#	print(names_[feat])
		#	print('Subjects with abnormal features', set(prob_subs2))

		if ((len(prob_subs3) > 0)):

			print(names_[feat])
			print('Subjects with abnormal features (NaN)', set(prob_subs3))


def remove_NaNs(feat_matrix_):
	a = ~np.any(np.isnan(feat_matrix_), axis = (0, 1, 2))
	print(a.shape)	
	feat_matrix_ = feat_matrix_[:, :, :, ~np.any(np.isnan(feat_matrix_), axis = (0, 1, 2))]

	return feat_matrix_

def remove_complete0(feat_matrix_):
	'''
	Removes all 0 columns from feat_matrix 	
	'''
	a = ~np.all(feat_matrix_ == 0, axis = (0, 1, 2))
	print(a.shape)	
	feat_matrix_ = feat_matrix_[:, :, :, ~np.all(feat_matrix_ == 0, axis = (0, 1, 2))]

	return feat_matrix_

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
