import argparse

import os
import sys
import glob

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import pickle

import numpy as np

from scipy import stats

from pylab import rcParams
rcParams['figure.figsize'] = 10, 9

# Testing settings
parser = argparse.ArgumentParser(description='Accuracies boxplots')
parser.add_argument('--path', type=str, default=None, metavar='Path', help='csv files path')
args = parser.parse_args()

name = args.path.split('_')[-1].split('.')[0]

df = pd.read_csv(args.path, usecols=['None', 'Whitening', 'Baseline 1', 'Baseline 2'])

order_plot = ['None', 'Whitening', 'Baseline 1', 'Baseline 2']	
box = sns.boxplot(data = df, color= 'hotpink', width = 0.2, linewidth = 1.0, showfliers = False, order = order_plot)
box.tick_params(labelsize=18)	
box.set_xticklabels(box.get_xticklabels())
box.set_xlabel('Normalization', fontsize = 18)
handles, _ = box.get_legend_handles_labels()
box.legend(handles, ["None", "Whitening", "Baseline 1", "Baseline 2"])

if name == 'covariateshift':
	box.set_ylabel('Estimated marginal shift', fontsize = 18)
if name == 'condshift':
	box.set_ylabel('Estimated conditional shift', fontsize = 18)	
elif name == 'classification':
	box.set_ylabel('Classification accuracy', fontsize = 18)

plt.grid(True, alpha = 0.3, linestyle = '--')
plt.savefig(name+'_boxplot.png')
plt.show()

