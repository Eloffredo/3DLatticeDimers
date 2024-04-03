import numpy as np
from argparse import ArgumentParser
import sys, os

parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-id1", "--sequenceid1", type = int, help="Specify the sequence ID of the first monomer")
parser.add_argument("-id2", "--sequenceid2", type = int, help="Specify the sequence ID of the second monomer")
args = parser.parse_args()

s1,s2 = args.sequenceid1, args.sequenceid2


if os.path.isfile(f'./src/tmp_contact_int_{s1}_{s2}.txt'):
    Cint = np.loadtxt(f'./src/tmp_contact_int_{s1}_{s2}.txt',dtype='int', delimiter=',')
elif os.path.isfile(f'./tmp_contact_int_{s1}_{s2}.txt'):
    Cint = np.loadtxt(f'./tmp_contact_int_{s1}_{s2}.txt',dtype='int', delimiter=',')                 
else: 
    file = open(f'./tmp_contact_int_{s1}_{s2}.txt','w')
    All_faces = np.loadtxt('./allfaces10000.dat',dtype='int')
    for f_ in range(6):
        for sf_ in range(6):
            for r in range(4):
    
                x = np.array([[All_faces[24*s1+4*f_+r][i]+1,All_faces[24*s2+4*sf_+0][i]+1] for i in range(3,12)],dtype=np.int16)
                new_column = np.array([(24*f_ + 4*sf_ + r +1) for _ in range(9)])
                x = np.insert(x, 0, new_column, axis=1)
                np.savetxt(file,x, delimiter = ' ',fmt='%d')
    file.close()    