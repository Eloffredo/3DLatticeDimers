# Design of 3D Lattice Protein Dimer Sequences using MCMC algorithms

This repository provides scripts to generate MSA and Evolutionary trajectories of 3-dimensional cubic lattice proteins dimers, undergoing monomeric or dimeric evolution in presence of selective pressures for stability and interaction. 
The script considers $\mathcal{N} = 10000$ cubic structures among the all possible $3 \times 3 \times 3$ possible ones for computational efficiency; they are entirely listed in the file ```src/allfacs10000.dat```. An example of lattice dimer involving two fixed structure in conctact across one face is showed in [lattice dimer](https://github.com/Eloffredo/3DLatticeDimers/src/lattice_dimer.pdf)

A demo notebook shows some dimer sequences in the MSA and the evolutionary trajectories of native stability fitnesses and interacting fitness $\mathcal{P}$.
The code in ```script_MSA.py``` starts from a dimer MSA or from random sequences and proposes mutation to generate a final MSA that statisfies the constraints imposed by the two selection pressures.
The code in ```script_EVO.py``` starts from a dimer MSA or from random sequences and tracks the fitnesses' evolution until they reach a steady state.

### Input files
Both script use as (optional) input file the MSA for both structures of the dimer. If they are not provided, random starting sequences are used. Starting MSAs should be placed under ```data/id.txt```, with the ID number of the structure chosen.
A conctact map for the interaction modes should be provided and can be generated using the script ```python3 ./src/contact_interaction_map.py -id1 ID1 -id2 ID2``` using the ID of the two structures forming the dimer.

### Options
The full list of options can be obtained by running ```python3 ./script_name.py --help```. In particular
* ```-t``` sets the mutational protocol $T$ during evolution. Use $0,1$ to mutate only the left or right structure of the dimer (monomeric evolution); use $2,3$ to propose mutation on both structures at the same time or once at time.
* ```-b``` sets the strength of the native selection pressure $\beta$. The higher this values the more the native stability of each structure will be prioritized over interaction.
* ```-g``` sets the strength of the native selection pressure $\gamma$. The higher this values the more the interaction between structures will be favored over single structure stability.
* ```-id``` is used to set the ID number to fix the cubic structure.

### Dependencies
The script are entirely written in python 3 and use standard scientific computing packages. Numba package must be installed in the environment for parallelization. 

### Example
The command lines to generate MSA and Evolutionary trajectories of a given dimer is the following

``` $ python3 ./src/contact_interaction_map.py -id1 624 -id2 6182 ``` to create the contact interaction map

``` $ python3 ./script_EVO.py -id1 624 -id2 6182 -b 1.0 -g 2.0 ``` to generate the evolutionary trajectories 

``` $ python3 ./script_MSA.py -id1 624 -id2 6182 -b 1.0 -g 2.0 -t 2 ``` to produce a MSA of interacting structures

### How to cite this repo:

If you use this repo for an academic publication, consider citing the following paper where those dimers are studied:

E. Loffredo et al, Evolutionary dynamics of a lattice dimer: a toy model for stability vs. affinity trade-offs in proteins, 2023 *J. Phys. A: Math. Theor. 56 455002*

If you have any question or comment, feel free to write me [here](mailto:emanuele.loffredo@phys.ens.fr).
