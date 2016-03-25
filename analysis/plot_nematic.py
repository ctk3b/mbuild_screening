import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')




plt.xlabel(r'$\gamma$ (s$^{-1}$)')
plt.ylabel('radius of gyration (nm)')


plt.savefig('rg_dist_{}K.pdf'.format(temp), bbox_inches='tight')
