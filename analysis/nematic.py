from glob import glob
import os
import pickle

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import seaborn as sns

sns.set_style('whitegrid')

folders = glob('../100ns_forces/task_*')
S2 = dict()

for base_path in folders:
    name = next(x for x in os.listdir(base_path)
                if x.startswith('alkane_') and x.endswith('.gro'))

    name = os.path.splitext(name)[0]
    S2[name] = dict()
    info = name.split('_')

    n_chains_per_surface = int(info[1][2:])
    hydrogens_per_surface = 100 - n_chains_per_surface

    try:
        traj = md.load(os.path.join(base_path, 'shear_whole.xtc'),
                       top=os.path.join(base_path, 'nvt.gro'))
    except FileNotFoundError:
        continue

    dt = traj.time[1] - traj.time[0]
    toss = 1000/dt  # 1 ns
    traj = traj[toss:]

    # Nematic order parameter
    atoms_per_chain = (traj.n_atoms // 2 - 1800 - hydrogens_per_surface) // n_chains_per_surface
    bot_chain_indices = [[n+x for x in range(atoms_per_chain)]
                         for n in range(1800,
                                        traj.n_atoms // 2 - hydrogens_per_surface,
                                        atoms_per_chain)]
    top_chain_indices = [[n+x for x in range(atoms_per_chain)]
                         for n in range(traj.n_atoms // 2 + 1800,
                                        traj.n_atoms - hydrogens_per_surface,
                                        atoms_per_chain)]

    bot_s2 = md.compute_nematic_order(traj, indices=bot_chain_indices)
    top_s2 = md.compute_nematic_order(traj, indices=top_chain_indices)

    plt.plot(traj.time / 1000, bot_s2, alpha=0.5, lw=0.5, label='bot')
    plt.plot(traj.time / 1000, top_s2, alpha=0.5, lw=0.5, label='top')
    plt.ylim(0, 1)
    plt.xlabel('time (ns)')
    plt.ylabel('S2')
    plt.legend()

    plt.savefig('{}.png'.format(name), bbox_inches='tight')
    plt.clf()

    S2[name]['top'] = np.mean(top_s2)
    S2[name]['top_std'] = np.std(top_s2)
    S2[name]['bot'] = np.mean(bot_s2)
    S2[name]['bot_std'] = np.std(bot_s2)
    print(name, traj.n_frames)
    del traj

with open('nematic_order.pickle', 'wb') as fh:
    pickle.dump(S2, fh)

