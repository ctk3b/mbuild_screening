import os
import MDAnalysis as mda
import numpy as np

base_path = "/Users/summeraz/Documents/Mixed_Monolayers/Pure_Alkane/H18"

trr = mda.coordinates.TRR.TRRReader(os.path.join(base_path, 'const_vel.trr'))

fric = []

skip = 500 # skip first 500ps

for frame in trr:
    if frame.time > 500:
        top_forces = frame.forces[:int(frame.n_atoms/2)]
        fric.append(np.sum(top_forces[:,0]))

nf = 1000.0 #kJ/mol/angstom
cof = np.mean(fric)/1000.0
print cof
