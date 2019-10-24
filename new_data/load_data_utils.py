import mne
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import read_layout
from mne.datasets import eegbci

import numpy as np

def load_subject_session(sub = 2, session_type = 'task'):

	subject = sub
	tmin, tmax = -1., 4.
	
	if session_type == 'task':
		event_id_annot = dict(T1=2, T2=3)	
		event_id_epochs = dict(hands=2, feet=3)
		runs = [6, 10, 14] # motor imagery: hands vs feet
	elif session_type == 'baseline1':
		event_id_annot = dict(T0=0)	
		event_id_epochs = dict(close=0)
		runs = 1
	elif session_type == 'baseline2':
		event_id_annot = dict(T0=0)	
		event_id_epochs = dict(open_=0)
		runs = 2		

	raw_fnames = eegbci.load_data(subject, runs, verbose=False)
	raws = [read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
	raw = concatenate_raws(raws)

	# strip channel names of "." characters
	raw.rename_channels(lambda x: x.strip('.'))

	# Apply band-pass filter
	raw.filter(0.5, 40., fir_design='firwin', skip_by_annotation='edge', verbose=False)

	events, _ = events_from_annotations(raw, event_id=event_id_annot, chunk_duration=1., verbose=False)

	#print(np.round((events[:, 0] - raw.first_samp) / raw.info['sfreq'], 3))


	picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
		             exclude='bads')
	#picks = mne.pick_channels(raw.info['ch_names'], ['C5', 'C3', 'Cz', 'C4', 'C6', 'T9', 'T10', 'Af7', 'Fpz', 'Af8', 'F5', 'F3', 'Fz', 'F4', 'F6'])


	# Read epochs (train will be done only between 1 and 2s)
	# Testing will be done with a running classifier
	epochs = Epochs(raw, events, event_id_epochs, tmin, tmax, proj=True, picks=picks,
		          baseline=None, preload=True, verbose=False)
	#epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
	labels = epochs.events[:, -1] - 2

	#epochs_data = epochs_train.get_data()

	psds, freqs = mne.time_frequency.psd_welch(epochs, fmin=0.5, fmax=30., verbose=False)

	# Normalize the PSDs
	psds /= np.sum(psds, axis=-1, keepdims=True)

	# specific frequency bands
	FREQ_BANDS = {"delta": [0.5, 4.5],
		      "theta": [4.5, 8.5],
		      "alpha": [8.5, 11.5],
		      "sigma": [11.5, 15.5],
		      "beta": [15.5, 30]}

	X = []
	for fmin, fmax in FREQ_BANDS.values():
		psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
		X.append(np.expand_dims(psds_band.reshape(len(psds), -1), axis=-1))

	X = np.concatenate(X, axis=-1)
	X = X.reshape(X.shape[0], X.shape[1]*X.shape[-1])
	return X, labels
	
def build_feat_matrix(subs, session_type, n_valid_samples):
	feat_dict_train = {}
	label_dict_train = {}
	feat_dict_valid = {}
	label_dict_valid = {}
	
	for sub in subs:	
		X, y = load_subject_session(sub, session_type)
		
		total_samples = X.shape[0]
		
		train_samples_idx = np.random.choice(range(total_samples), size=total_samples-n_valid_samples, replace=False)
		valid_samples_idx = [] 

		for idx in range(total_samples):	
			if idx not in train_samples_idx:
				valid_samples_idx.append(idx)
		
		feat_dict_train[sub-1] = X[train_samples_idx]
		label_dict_train[sub-1] = y[train_samples_idx]
		feat_dict_valid[sub-1] = X[valid_samples_idx]
		label_dict_valid[sub-1] = y[valid_samples_idx]
	
	return feat_dict_train, label_dict_train, feat_dict_valid, label_dict_valid	
	
if __name__ == '__main__':
	X_, y_ = load_subject_session(2, 'task')
	print(X_.shape)
	print(y_.shape)
	
	feat_train, label_train, feat_valid, label_valid = build_feat_matrix(range(1, 2+1), 'task', 30)
	
