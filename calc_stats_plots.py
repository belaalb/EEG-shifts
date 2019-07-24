import argparse
import os
import sys
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Testing settings
parser = argparse.ArgumentParser(description='Accuracies boxplots')
parser.add_argument('--train', type=str, default=None, metavar='Path', help='csv files path')
parser.add_argument('--test', type=str, default=None, metavar='Path', help='csv files path')
parser.add_argument('--avgcondshift', type=str, default=None, metavar='Path', help='csv files path')
args = parser.parse_args()

df1 = pd.read_csv(args.train, usecols=['None', 'Whitening', 'Baseline 1', 'Baseline 2'])
df2 = pd.read_csv(args.test, usecols=['None', 'Whitening', 'Baseline 1', 'Baseline 2'])

data1 = df1.values
data2 = df2.values

diff = np.abs(data1 - data2)

subs = []

# Boxplot estimated generalization gap per subject 
for sub in range(9):
	s = diff[sub:-1:9, :]
	subs.append(pd.DataFrame(s[:, 0:2], columns=['None', 'Whitening']).assign(Subject=sub))
	print('{} avg:'.format(sub), s.mean(0))
	print('{} std:'.format(sub), s.std(0))

s = diff[0:-1:1, :]
subs.append(pd.DataFrame(s[:, 0:2], columns=['None', 'Whitening']).assign(Subject='All'))
print('All avg:', s.mean(0))
print('All std:', s.std(0))

df_subs = pd.concat(subs)
mdf = pd.melt(df_subs, id_vars=['Subject'], var_name=['Normalization']) 

ax = sns.boxplot(x="Subject", y="value", hue="Normalization", color= 'hotpink', data=mdf, width = 0.5, linewidth = 0.7, showfliers = False)  
ax.set_ylabel('Estimated generalization gap', fontsize = 10)	
plt.grid(True, alpha = 0.3, linestyle = '--') 
plt.savefig('gengap_boxplot.png')
plt.show()


# Plot average conditional shift per subject
df = pd.read_csv(args.avgcondshift, usecols=['0', '1', '2', '3', '4', '5', '6', '7', '8', 'Normalization'])
mdf = pd.melt(df, id_vars=['Normalization'], var_name=['Subject']) 

ax = sns.catplot(x="Subject", y="value", hue="Normalization", color= 'hotpink', edgecolor=".5", data=mdf, kind="bar", errcolor='gray', errwidth=0.3, capsize=0.1)    
ax.set(ylabel='Cross-subject disparity')
plt.grid(True, alpha = 0.3, linestyle = '--') 
plt.savefig('cond_shift_persub.png')
plt.show()
