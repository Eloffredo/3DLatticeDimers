import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('./src/')
sys.path.append('./data')
import utils_prot as util
import MC_tools as MC
from numba import njit,prange,jit
import time
from argparse import ArgumentParser

DATA_PATH = './data/'

#Args parser 
parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-id1", "--sequenceid1", type = int, help="Specify the sequence ID of the first monomer")
parser.add_argument("-id2", "--sequenceid2", type = int, help="Specify the sequence ID of the second monomer")
parser.add_argument("-p", "--path", default = DATA_PATH, help="Specify the full path to folder to retrieve and store data")
parser.add_argument("-t", "--typeofmut",  type = int, default=2, help="Specify if monomeric or dimeric mutational protocol is desired")
parser.add_argument("-b", "--beta", default=1.0, help="Specify single monomer strength")
parser.add_argument("-g", "--gamma", default=1.0, help="Specify dimer interaction strength")
parser.add_argument("-Navg", "--Numaverage", type = int, default=15000, help="Specify the total number sequences to consider")
parser.add_argument("-Nit", "--Numiter", type = int, default=3000, help="Specify the total number of iterations to perform")

args = parser.parse_args()

s1,s2 = args.sequenceid1, args.sequenceid2
T = args.typeofint
DATA_PATH = args.path

assert T in [0,1,2,3], "Invalid value for Interaction Type. Please specify 0,1 for monomeric evolution or 2,3 for dimeric evolution" 

Naverage = args.Numaverage
Numit = args.Numiter
beta, gammaint = args.beta, args.gamma
L = 27


PATH_MSA_ONE, PATH_MSA_TWO = DATA_PATH+f'{s1}.txt',   DATA_PATH+f'{s2}.txt'

if os.path.isfile(PATH_MSA_ONE) and os.path.isfile(PATH_MSA_TWO):
    with open(PATH_MSA_ONE, 'r') as file:
        list_sequences1 = [line.strip() for line in file.readlines()]
        seqvec_1 = np.array([util.sqlton(list_sequences1[k]) for k in range(len(list_sequences1))],dtype=np.int8)

    with open(PATH_MSA_TWO, 'r') as file:
        list_sequences2 = [line.strip() for line in file.readlines()]
        seqvec_2 = np.array([util.sqlton(list_sequences2[k]) for k in range(len(list_sequences2))],dtype=np.int8)
else: 
    seqvec_2 = np.random.randint(0,20,size = (Naverage,L),dtype=np.int8)                        
    seqvec_1 = np.random.randint(0,20,size = (Naverage,L),dtype=np.int8)
    
util.get_interaction_map(s1,s2)
                  
                        
OUTPUT_PATH = DATA_PATH + f'Full_MSA_beta_{beta}_gamma_{gammaint}_struct_a{s1}_b{s2}blob.txt'

start = time.time()
print('...',end = '\n')
output = MC.MSA_loop(seqvec_1,seqvec_2,beta,gammaint,T,Naverage,Numit,s1,s2)

file=open(OUTPUT_PATH,'w')
np.savetxt(file,output.reshape(Naverage,2*L), fmt = '%d', delimiter=" ")
file.close()

end = time.time()
tot_time = end - start
print( f'This loop with s1:{s1} and s2:{s2} took {tot_time} seconds \
to generate a MSA of depth {Naverage} with {Numit} steps.', end ='\n' )
