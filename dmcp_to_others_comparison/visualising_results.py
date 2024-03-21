# -*- coding: utf-8 -*-
"""
Comparing performance of various methods
========================================

Created on Thu Mar 21 10:39:26 2024

@author: thejasvi beleyur
Code released under MIT License

"""
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
results_files = glob.glob('mic-to-cave-results\\'+'*.csv')
all_results = pd.concat([pd.read_csv(each) for each in results_files]).reset_index(drop=True)
all_results['method'] = all_results['method-date'].apply(lambda X: X.split('_')[1])
all_results['method'] = pd.Categorical(all_results['method'],
                                       ["dmcp", "disk", "r2d2","sift","sosnet","superpoint"])

by_method = all_results.groupby('method')

distances = []
names = []
for method, subdf in by_method:
    distances.append(subdf['predist'].to_numpy())
    #distances.append(subdf['postdist'].to_numpy())
    names.append(method)
plt.figure()
plt.violinplot(distances)
plt.xticks(range(1,len(distances)+1),names)

median_predist = by_method.aggregate(np.nanmedian)['predist']


#%%
plt.figure(figsize=(7,3))	
# boxplot with seaborn
sns.boxplot(x = "method", y = "predist", data = all_results,
            boxprops=dict(facecolor="none"), fliersize=0.01, whis=[25,75]) 
sns.stripplot(x='method', y='predist', data=all_results,  edgecolor='none', jitter=True,
              alpha=0.5, size=4, )

plt.ylabel('Distance to nearest\n mesh point, m', fontsize=11, labelpad=-6);

plt.xlabel('Method name', fontsize=11, labelpad=0);
sns.despine(); plt.yscale('log')

plt.yticks(ticks=[1e-3, 1e-2, 1e-1, 5e-1, 1, 1e1, 1e2, 1e3],
           labels=[0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000],
           fontsize=9)
plt.tight_layout()
plt.savefig('preicp_mic-to-cave-comparison.png')
plt.savefig('preicp_mic-to-cave-comparison.eps')


