import os

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd
from pymbar import timeseries
import seaborn as sns
from scipy.stats import linregress
from scipy import signal

sns.set_style('whitegrid')

base_path = '../output/task_0'
traj = md.load(os.path.join(base_path, 'shear.xtc'),
               top=os.path.join(base_path, 'nvt.gro'))

dt = traj.time[1] - traj.time[0]
toss = 1000/dt  # 1 ns

t = traj.time[toss:]
x = traj.xyz[toss:, traj.n_atoms/2 + 2, 0]

box_x = traj.unitcell_lengths[0, 0]

n_hops = np.zeros_like(x)
for i, p in enumerate(x):
    if p - x[i-1] > 0.5*box_x:
        n_hops[i:] += 1
    elif p - x[i-1] < -0.5*box_x:
        n_hops[i:] -= 1

x -= n_hops * box_x

# plt.plot(t, x)
# plt.xlabel('time (ps)')
# plt.ylabel('x position (nm)')

slope, intercept, r, p, std_err = linregress(t, x)

x -= (slope * t + intercept)

#t0, g, Neff_max = timeseries.detectEquilibration(x)
#x_equlibrated = x[t0:]
#plt.axvline(t[t0], color='red')

# plt.plot(t, x)
# plt.xlabel('time (ps)')
# plt.ylabel('x position (nm)')


ft = np.fft.rfft(x)
freq = np.fft.rfftfreq(len(x), d=dt)
plt.plot(freq[2:20]*1e3, np.real(ft[2:20])**2)
plt.xlabel('GHz')
plt.ylabel('amplitude')
plt.savefig('alkane_n-100_l-6_fourier.pdf', bbox_inches='tight')
