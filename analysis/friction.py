from glob import glob
import os
import pickle

import matplotlib.pyplot as plt
import mdtraj as md
import MDAnalysis as mda
import numpy as np
import seaborn as sns

sns.set_style('whitegrid')

folders = glob('../monolayers/output/task_*')
forces = dict()

for base_path in folders:
    name = next(x for x in os.listdir(base_path)
                if x.startswith('alkane_') and x.endswith('.gro'))

    name = os.path.splitext(name)[0]
    info = name.split('_')

    try:
        trr = mda.coordinates.TRR.TRRReader(os.path.join(base_path, 'shear.trr'))
    except IOError:
        continue

    # dt = traj.time[1] - traj.time[0]
    # toss = 1000/dt  # 1 ns
    # traj = traj[toss:]

    fric = []
    toss = 1000 # skip first 1ns
    for frame in trr:
        if frame.time > toss:
            top_forces = frame.forces[:int(frame.n_atoms/2)]
            fric.append(np.sum(top_forces[:, 0]))
        last_time = frame.time

    normal_force = 10000.0 #kJ/mol/angstom
    cof = np.mean(fric) / normal_force

    plt.plot(np.linspace(1000.0, last_time, len(fric)) / 1000, fric)
    #plt.ylim(0, 1)
    plt.xlabel('time (ns)')
    plt.ylabel('Shear force (kJ/(mol nm)')

    plt.savefig('{}_shear_force.png'.format(name), bbox_inches='tight')
    plt.clf()

    forces[name] = np.asarray(fric)
    print(name, last_time)
    del trr

with open('forces.pickle', 'wb') as fh:
    pickle.dump(forces, fh)

