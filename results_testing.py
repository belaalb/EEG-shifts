import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


csv_name = './Results/all_normalizations_classification.csv'
df = pd.read_csv(csv_name, usecols=['no', 'white', 'bl1', 'bl2'])
results_dict = df.to_dict('list')


p_no_white = stats.mannwhitneyu(results_dict['no'], results_dict['white'])
print('No vs White normalization', p_no_white)
p_no_bl1 = stats.mannwhitneyu(results_dict['no'], results_dict['bl1'])
print('No vs BL1 normalization', p_no_bl1)
p_no_bl2 = stats.mannwhitneyu(results_dict['no'], results_dict['bl2'])
print('No vs Bl2 normalization', p_no_bl2)

p_white_bl1 = stats.mannwhitneyu(results_dict['white'], results_dict['bl1'])
print('White vs BL1 normalization', p_white_bl1)
p_white_bl2 = stats.mannwhitneyu(results_dict['white'], results_dict['bl2'])
print('No vs BL1 normalization', p_white_bl2)

p_bl1_bl2 = stats.mannwhitneyu(results_dict['bl1'], results_dict['bl2'])
print('BL1 vs BL2 normalization', p_bl1_bl2)
