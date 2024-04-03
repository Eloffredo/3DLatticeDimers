import numpy as np
from numba import jit,njit,prange
import utils_prot as util
import sys, os
sys.path.append('./src')


Ncont =28
Nstruct =10000
Ninterface = 144
Ncontface=9

sface=0

@jit(nopython=True)
def step_L(seq1,seq2,beta,gammaint,s1,s2):
    
    v=np.random.randint(0,27)
    newseq1=np.copy(seq1)
    newseq1[v] = np.random.randint(0,20) 
    
    test=((util.pnat(s1, newseq1)/util.pnat(s1, seq1))**beta)*((util.pbind(sface,newseq1,seq2)/util.pbind(sface,seq1,seq2))**gammaint)
    
    if (test>1): 
        seq1=np.copy(newseq1)
       
    else:
        if (np.random.uniform(0,1)<= (test**(1000)) ):
            seq1=np.copy(newseq1)
               
    return (seq1,seq2)


@jit(nopython=True)
def step_R(seq1,seq2,beta,gammaint,s1,s2):
    
    v=np.random.randint(0,27)
    newseq2=np.copy(seq2)
    newseq2[v] = np.random.randint(0,20) 
    
    test=((util.pnat(s2, newseq2)/util.pnat(s2, seq2))**beta)*((util.pbind(sface,seq1,newseq2)/util.pbind(sface,seq1,seq2))**gammaint)
    
    if (test>1): 
        seq2=np.copy(newseq2)
       
    else:
        if (np.random.uniform(0,1)<= (test**(1000)) ):
            seq2=np.copy(newseq2)
               
    return (seq1,seq2)

@jit(nopython=True)
def step_both(seq1,seq2,beta,gammaint,s1,s2):
    
    v=np.random.randint(0,27)
    newseq1=np.copy(seq1)
    newseq1[v] = np.random.randint(0,20) 
    u=np.random.randint(0,27)
    newseq2=np.copy(seq2)
    newseq2[u] = np.random.randint(0,20) 
    
    test=((util.pnat(s1, newseq1)/util.pnat(s1, seq1))**beta)*((util.pnat(s2, newseq2)/util.pnat(s2, seq2))**beta)*((util.pbind(sface,newseq1,newseq2)/util.pbind(sface,seq1,seq2))**gammaint)
    
    if (test>1): 
        seq1=np.copy(newseq1)
        seq2=np.copy(newseq2)
       
    else:
        if (np.random.uniform(0,1)<= (test**(1000)) ):
            seq1=np.copy(newseq1)
            seq2=np.copy(newseq2)
               
    return (seq1,seq2)

@jit(nopython=True)
def step_LR(seq1,seq2,beta,gammaint,s1,s2):
    
    value =np.array([1,2],dtype=np.int8)
    dec_value = np.random.choice(value)
    
    if (dec_value ==1):
        newseq2=np.copy(seq2)
        v=np.random.randint(0,27)
        newseq1=np.copy(seq1)
        newseq1[v] = np.random.randint(0,20)
    elif (dec_value ==2):
        newseq1=np.copy(seq1)
        u=np.random.randint(0,27)
        newseq2=np.copy(seq2)
        newseq2[u] = np.random.randint(0,20) 
    
    test=((util.pnat(s1, newseq1)/util.pnat(s1, seq1))**beta)*((util.pnat(s2, newseq2)/util.pnat(s2, seq2))**beta)*((util.pbind(sface,newseq1,newseq2)/util.pbind(sface,seq1,seq2))**gammaint)
    
    if (test>1): 
        seq1=np.copy(newseq1)
        seq2=np.copy(newseq2)
       
    else:
        if (np.random.uniform(0,1)<= (test**(1000)) ):
            seq1=np.copy(newseq1)
            seq2=np.copy(newseq2)
               
    return (seq1,seq2)


@jit(nopython=True)
def metropolis(seq1,seq2,beta,gammaint,I,Numit,s1,s2):
    
    if(I==0):
     
        for i in range(Numit):
            seq1,seq2 = step_L(seq1,seq2,beta,gammaint,s1,s2)
        return [seq1,seq2]

    elif(I==1):
     
        for i in range(Numit): 
            seq1,seq2 = step_R(seq1,seq2,beta,gammaint,s1,s2)       
        return [seq1,seq2]
    
    elif(I==2):
     
        for i in range(Numit):
            seq1,seq2 = step_both(seq1,seq2,beta,gammaint,s1,s2)        
        return [seq1,seq2]
    
    elif(I==3):
     
        for i in range(Numit):
            seq1,seq2 = step_LR(seq1,seq2,beta,gammaint,s1,s2)       
        return [seq1,seq2]

@njit(parallel = True)
def MSA_loop(seqvec_1,seqvec_2,beta,gammaint,I,Naverage,Numit,s1,s2):
    
    MSA_ = np.zeros((Naverage,2,27),dtype="int")
    for k in prange(Naverage):

        seq1=seqvec_1[k]; seq2=seqvec_2[k]

        out =metropolis(seq1,seq2,beta,gammaint,I,Numit,s1,s2)
        MSA_[k][0] = out[0]
        MSA_[k][1] = out[1]        

    return MSA_


@jit(nopython=True)
def metropolis_new(seq1,seq2,beta,gammaint,I,Numit,s1,s2):
    
    if(I==0):
        
        data = np.zeros((Numit,4),dtype=np.float64)
        for i in range(Numit):

            data[i] = float(i),util.pnat(s1,seq1),util.pnat(s2,seq2),util.pbind(sface,seq1,seq2)
            seq1,seq2 = step_L(seq1,seq2,beta,gammaint,s1,s2)
        return (data)
    
    if(I==1):
        
        data = np.zeros((Numit,4),dtype=np.float64)
        for i in range(Numit):

            data[i] = float(i),util.pnat(s1,seq1),util.pnat(s2,seq2),util.pbind(sface,seq1,seq2)
            seq1,seq2 = step_R(seq1,seq2,beta,gammaint,s1,s2)
        return (data)
    
    
    if(I==2):
        
        data = np.zeros((Numit,4),dtype=np.float64)
        for i in range(Numit):

            data[i] = float(i),util.pnat(s1,seq1),util.pnat(s2,seq2),util.pbind(sface,seq1,seq2)
            seq1,seq2 = step_both(seq1,seq2,beta,gammaint,s1,s2)
        return (data)
    
    if(I==3):
        
        data = np.zeros((Numit,4),dtype=np.float64)
        for i in range(Numit):

            data[i] = float(i),util.pnat(s1,seq1),util.pnat(s2,seq2),util.pbind(sface,seq1,seq2)
            seq1,seq2 = step_LR(seq1,seq2,beta,gammaint,s1,s2)
        return (data)
    
    
@njit(parallel=True)
def trajectory_loop(seqvec_1,seqvec_2,beta,gammaint,I,Naverage,Numit,s1,s2):
      
    data=np.zeros((Naverage*Numit,4),dtype=np.float64)
    for k in prange(Naverage):
       
        seq1=seqvec_1[k]; seq2=seqvec_2[k]
        data[k*Numit:(k+1)*Numit] = metropolis_new(seq1,seq2,beta,gammaint,I,Numit,s1,s2)
    return data