import itertools as it
import time
import os
import shutil
import textwrap

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import seaborn as sns

import mbuild as mb
import metamds as mds


used_random_patterns = dict()

def build_monolayer(chain_length, n_molecules, pattern_class, **kwargs):
    from mbuild.examples import AlkaneMonolayer
    if pattern_class is mb.Random2DPattern:
        if n_molecules in used_random_patterns:
            pattern = used_random_patterns[n_molecules]
        else:
            pattern = pattern_class(n_molecules)
        pattern_name = 'rand'
    if pattern_class is mb.Grid2DPattern:
        pattern = pattern_class(int(np.sqrt(n_molecules)), int(np.sqrt(n_molecules)))
        pattern_name = 'grid'

    bot = AlkaneMonolayer(pattern, tile_x=1, tile_y=1, chain_length=chain_length)
    mb.translate(bot, [0, 0, 2])

    bot_box = bot.boundingbox
    bot_of_bot = bot_box.mins[2]

    bot_rigid = [i + 1 for i, a in enumerate(bot.particles())
                 if (a.pos[2] < bot_of_bot + 0.05) and a.name == 'Si']
    n_particles = bot.n_particles
    top_rigid = [i + n_particles for i in bot_rigid]

    top = mb.clone(bot)
    mb.spin_y(top, np.pi)
    top_of_bot = bot_box.maxs[2]
    bot_of_top = top.boundingbox.mins[2]
    mb.translate(top, [0, 0, top_of_bot - bot_of_top + 0.5])

    monolayer = mb.Compound([bot, top])
    monolayer.name = 'alkane_n-{}_l-{}-{}'.format(n_molecules, chain_length, pattern_name)
    rigid_groups = {'bot': bot_rigid,
                    'top': top_rigid}
    return monolayer, rigid_groups


def create_run_script(build_func, forcefield, input_dir, **kwargs):
    compound, rigid_groups = build_func(**kwargs)
    name = compound.name
    em = os.path.join(input_dir, 'em.mdp')
    nvt = os.path.join(input_dir, 'nvt.mdp')
    shear = os.path.join(input_dir, 'const_vel.mdp')
    gro = '{name}.gro'.format(name=name)
    top = '{name}.top'.format(name=name)
    ndx = '{name}.ndx'.format(name=name)

    box = compound.boundingbox
    compound.periodicity += np.array([0, 0, 5 * box.lengths[2]])
    compound.save(top, forcefield=forcefield, overwrite=True)

    with open(ndx, 'w') as f:
        f.write('[ System ]\n')
        atoms = '{}\n'.format(' '.join(str(x + 1) for x in range(compound.n_particles)))
        f.write(textwrap.fill(atoms, 80))
        f.write('\n')
        for name, indices in rigid_groups.items():
            f.write('[ {name} ]\n'.format(name=name))
            atoms = '{}\n'.format(' '.join(str(x) for x in indices))
            f.write(textwrap.fill(atoms, 80))
            f.write('\n')

    cmds = list()
    cmds.append('gmx grompp -f {mdp} -c {gro} -p {top} -n {ndx} -o em.tpr'.format(mdp=em, gro=gro, top=top, ndx=ndx))
    cmds.append('gmx mdrun -v -deffnm em -ntmpi 1')

    cmds.append('gmx grompp -f {mdp} -c em.gro -p {top} -n {ndx} -o nvt.tpr'.format(mdp=nvt, top=top, ndx=ndx))
    cmds.append('gmx mdrun -v -deffnm nvt -ntmpi 1')

    cmds.append('gmx grompp -f {mdp} -c nvt.gro -p {top} -n {ndx} -o shear.tpr'.format(mdp=shear, top=top, ndx=ndx))
    cmds.append('gmx mdrun -v -deffnm shear -ntmpi 1')

    # add shearing commands

    return cmds

if __name__ == '__main__':
    # Initialize a simulation instance with a template and some metadata
    try:
        shutil.rmtree('output')
    except FileNotFoundError:
        pass
    sim = mds.Simulation(name='monolayer', template=create_run_script, output_dir='output')

    chain_lengths = [6, 9, 12, 15, 18, 21]
    n_molecules = [100, 81, 64]
    # patterns = [mb.Random2DPattern, mb.Grid2DPattern]

    chain_lengths = [6]
    n_molecules = [100]
    patterns = [mb.Random2DPattern]
    for length, n_mols, pattern in it.product(chain_lengths, n_molecules, patterns):
        if n_mols == 100 and pattern is mb.Grid2DPattern:
            continue
        parameters = {'chain_length': length,
                      'n_molecules': n_mols,
                      'forcefield': 'OPLS-aa',
                      'pattern_class': pattern,
                      'build_func': build_monolayer}

        # Parameterize our simulation template
        sim.parametrize(**parameters)

    # Run
    sim.execute_all(hostname='rahman.vuse.vanderbilt.edu', username='ctk3b')
    # sim.execute_all()

    # Analyze
    # trajectories = task.get_output_files('trajectories')
    # topologies = task.get_output_files('topologies')
    # Pick which one to select?
    import ipdb; ipdb.set_trace()

    # trj_path = os.path.join(task.output_dir, 'nvt.xtc')
    # top_path = os.path.join(task.output_dir, 'em.gro')
    # traj = md.load(trj_path, top=top_path)
    # print(traj)

    # RDF
    # pairs = traj.top.select_pairs('name C', 'name C')
    # r, g_r = md.compute_rdf(traj, pairs)
    # plt.plot(r, g_r)
    # plt.xlabel('r (nm)')
    # plt.ylabel('g(r)')
    # plt.show()
    #
    # s2 = md.compute_nematic_order(traj, 'residues')
    # plt.plot(traj.time, s2)
    # plt.xlabel('time (ps)')
    # plt.ylabel('S2')
