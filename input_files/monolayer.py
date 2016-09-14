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

from mbuild.examples import AlkaneMonolayer, PegMonolayer, PegOxyCapMonolayer # needed to import somewhere up here or directly use mb.examples... elsewhere TJ
from mbuild.lib.surfaces import Betacristobalite
from mbuild.lib.bulk_materials import AmorphousSilica

used_random_patterns = dict()

def build_monolayer(chain_length, n_molecules, pattern_class, polymer_class=AlkaneMonolayer, 
                    surface_class=Betacristobalite, name=None, **kwargs):
    if pattern_class is mb.Random2DPattern:
        if n_molecules in used_random_patterns:
            pattern = used_random_patterns[n_molecules]
        else:
            pattern = pattern_class(n_molecules)
            used_random_patterns[n_molecules] = pattern
    if pattern_class is mb.Grid2DPattern:
        pattern = pattern_class(int(np.sqrt(n_molecules)), int(np.sqrt(n_molecules)))
    
    # --------------------------------------------------------------- # TJ
    """
    If elif statement to check surface type and create for each monolayer. TJ
    """
    if surface_class is Betacristobalite:
        surface = surface_class()
    elif surface_class is mb.SilicaInterface:
        surface = surface_class(bulk_silica=AmorphousSilica(), thickness=1.2)
    bot = polymer_class(pattern, surface=surface, tile_x=1, tile_y=1, chain_length=chain_length)
    # --------------------------------------------------------------- # TJ
    mb.translate(bot, [0, 0, 2])

    bot_box = bot.boundingbox
    bot_of_bot = bot_box.mins[2]

    bot_rigid = [i + 1 for i, a in enumerate(bot.particles())
                 if (a.pos[2] < bot_of_bot + 0.2) and a.name == 'Si']
    n_particles = bot.n_particles
    top_rigid = [i + n_particles for i in bot_rigid]

    top = mb.clone(bot)
    mb.spin_y(top, np.pi)
    top_of_bot = bot_box.maxs[2]
    bot_of_top = top.boundingbox.mins[2]
    mb.translate(top, [0, 0, top_of_bot - bot_of_top + 0.5])

    monolayer = mb.Compound([bot, top])
    
    if not name:
        name = '{}_n-{}_l-{}-{}-{}'.format(polymer_class.__name__[:3], n_molecules, chain_length,
                                           surface_class.__name__, pattern_class.__name__[:4]) 
    monolayer.name = name
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
    cmds.append('gmx mdrun -v -deffnm em -ntmpi 2')

    cmds.append('gmx grompp -f {mdp} -c em.gro -p {top} -n {ndx} -o nvt.tpr'.format(mdp=nvt, top=top, ndx=ndx))
    cmds.append('gmx mdrun -v -deffnm nvt -ntmpi 2')

    cmds.append('gmx grompp -f {mdp} -c nvt.gro -p {top} -n {ndx} -o shear.tpr'.format(mdp=shear, top=top, ndx=ndx))
    cmds.append('gmx mdrun -v -deffnm shear -ntmpi 2')

    # add shearing commands

    return cmds

def create_run_script_nersc(build_func, forcefield, input_dir, **kwargs):
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
    cmds.append('srun gmx_sp grompp -f {mdp} -c {gro} -p {top} -n {ndx} -o em.tpr'.format(mdp=em, gro=gro, top=top, ndx=ndx))
    cmds.append('srun gmx_sp mdrun -v -deffnm em')

    cmds.append('srun gmx_sp grompp -f {mdp} -c em.gro -p {top} -n {ndx} -o nvt.tpr'.format(mdp=nvt, top=top, ndx=ndx))
    cmds.append('srun gmx_sp mdrun -v -deffnm nvt')

    cmds.append('srun gmx_sp grompp -f {mdp} -c nvt.gro -p {top} -n {ndx} -o shear.tpr'.format(mdp=shear, top=top, ndx=ndx))
    cmds.append('srun gmx_sp mdrun -v -deffnm shear')

    # add shearing commands

    return cmds


if __name__ == '__main__':
    # Initialize a simulation instance with a template and some metadata
    try:
        shutil.rmtree('output')
    except FileNotFoundError:
        pass
    sim = mds.Simulation(name='monolayer', template=create_run_script_nersc, output_dir='output')

    # polymers = [AlkaneMonolayer, PegMonolayer] 
    polymers = [PegOxyCapMonolayer] # Added to change polymer on surface TJ
    surfaces = [Betacristobalite, mb.SilicaInterface] # Added to change the surface substrate TJ
    chain_lengths = [7, 10, 13, 16, 19, 22] # need to be greater than a multiple of 3 by one to account for the peg monolayers 3 atom monomer length and the methyl cap TJ
    n_molecules = [100]
    patterns = [mb.Random2DPattern]
    """
    Added in poly and surface to iteratives because we can change the polymers and surface substrate.
    poly=alkane and surface=crystalline for base/empty case. TJ 
    """

    for length, n_mols, pattern, poly, surface_class in it.product(chain_lengths, n_molecules, patterns, polymers, surfaces):
        if n_mols == 100 and pattern is mb.Grid2DPattern:
            continue
        
        name = '{}_n-{}_l-{}_{}_{}'.format(poly.__name__[:4], n_mols, length, 
                                           surface_class.__name__, pattern.__name__[:4]) 
        
        parameters = {'chain_length': length,
                      'n_molecules': n_mols,
                      'surface_class': surface_class,
                      'forcefield': 'OPLS-aa',
                      'pattern_class': pattern,
                      'polymer_class': poly,
                      'build_func': build_monolayer,
                      'name': name}

        # Parameterize our simulation template
        sim.parametrize(**parameters)
        # Send simulation data to a database
        parameters['normal_force(kJ/mol/A)'] = 0900.0
        sim.add_to_db(collection='peg_oxy_cap', 
                      trajectories=['shear.trr', 'shear.xtc', 'shear_whole.xtc'], **parameters)

    # Run
    # sim.execute_all(hostname='rahman.vuse.vanderbilt.edu', username='jonestj1')
    sim.execute_all(hostname='edison.nersc.gov', username='jonestj')
    # sim.execute_all(hostname='login.accre.vanderbilt.edu', username='jonestj1', resource="ACCRE")
    # sim.execute_all()
