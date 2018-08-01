# -*- coding: utf-8 -*-
"""
This script was just to generate the density scatters
on the basis of the raw data.
@author: vr308
"""


import matplotlib.pylab as plt
import ConfigParser
import numpy as np
import os
import warnings
import itertools
import sys
sys.path.append(os.path.join(os.path.realpath('..')))
settings = ConfigParser.ConfigParser()
settings.read('../settings.ini')

from higgs_data import HiggsData

def plot_hexbin(d, C, cmap, f):
    plt.hexbin(np.array(d[f[0]]), np.array(d[f[1]]), C=C, reduce_C_function=np.sum, bins='log', gridsize=200, cmap=cmap)
    plt.xlabel(f[0])
    plt.ylabel(f[1])
    plt.colorbar()

def multiplot_hexbin(b, s):
    f1 = 'DER_mass_transverse_met_lep'
    f2 = 'DER_mass_MMC'
    f3 = 'DER_mass_jet_jet'
    f4 = 'DER_pt_h'
    feature_pairs = list(itertools.combinations([f1,f2,f3,f4], 2))
    feature_pairs_ = list(itertools.chain(*zip(feature_pairs,feature_pairs)))
    b_clean = b[(b[f1] != -999.0) & (b[f2] != -999.0) & (b[f3] != -999.0) & (b[f4] != -999.0)]
    s_clean = s[(s[f1] != -999.0) & (s[f2] != -999.0) & (s[f3] != -999.0) & (s[f4] != -999.0)]
    plt.figure(figsize=(15,15))
    multiplot_range = xrange(6)
    for i, k in zip(multiplot_range, feature_pairs_[0:6]):
        plt.subplot(str(32)+str(i+1))
        if i%2:
            plot_hexbin(s_clean, s_clean['Weight'], plt.get_cmap('Reds'), k)
        else:
            plot_hexbin(b_clean, b_clean['Weight'], plt.get_cmap('Blues'), k)
    plt.tight_layout()
    filename = '../Graphs/scatter1.png'
    print 'Saving plot in ' + filename
    plt.savefig(filename)
    plt.figure(figsize=(15,15))
    multiplot_range = xrange(6)
    for i, k in zip(multiplot_range, feature_pairs_[6:12]):
        plt.subplot(str(32)+str(i+1))
        if i%2:
            plot_hexbin(s_clean, s_clean['Weight'], plt.get_cmap('Reds'), k)
        else:
            plot_hexbin(b_clean, b_clean['Weight'], plt.get_cmap('Blues'), k)
    plt.tight_layout()
    filename = '../Graphs/scatter2.png'
    print 'Saving plot in ' + filename
    plt.savefig(filename)


if __name__ == "__main__":

    print 'Loading data'
    hd = HiggsData(path = settings.get('paths','path_data'), imputation=False)
    b = hd.processed_input[hd.processed_input.Label == -1]
    s = hd.processed_input[hd.processed_input.Label == 1]

    warnings.filterwarnings('ignore')
    multiplot_hexbin(b, s)


plt.figure(figsize=(10,5))
plt.subplot(121)
plot_hexbin(b_clean, b_clean['Weight'], plt.get_cmap('Blues'), k)
plt.ylim(3,7)
plt.xlim(0,6)
plt.subplot(122)
plot_hexbin(s_clean, s_clean['Weight'], plt.get_cmap('Reds'), k)


plt.figure(figsize=(10,5))
plt.subplot(122)
plt.hist(classifier.test_posterior[:,1][Y_test == -1], bins=100, histtype='stepfilled', alpha=0.5)
plt.hist(classifier.test_posterior[:,1][Y_test == 1], bins=100, histtype='stepfilled', alpha=0.5, color='r')
plt.title('Class separation after learning', fontsize='small')
plt.vlines(x=np.percentile(classifier.test_posterior[:,1], 85), ymin=0, ymax=10000, color='r')
plt.xlabel('Tree Ensemble score', fontsize='small')
plt.subplot(121)
plt.hist(classifier.X_test['DER_mass_MMC'][(Y_test == -1) & (classifier.X_test.DER_mass_MMC != classifier.X_test.DER_mass_MMC[350014])], bins=100, histtype='stepfilled', alpha=0.5)
plt.hist(classifier.X_test['DER_mass_MMC'][Y_test == 1], bins=100, histtype='stepfilled',alpha=0.5, color='r')
plt.title('Class separation provided by mass feature', fontsize='small')
plt.xlabel('Derived mass MMC', fontsize='small')
