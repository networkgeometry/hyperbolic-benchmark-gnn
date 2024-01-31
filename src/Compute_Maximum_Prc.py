

#Run it by      python Compute_Maximum_Prc.py   config.yaml


import re
import os
import sys
from yaml import load, Loader
import itertools
import glob
import numpy as np
import pandas as pd 
import math

def Num_Removed_link(Pr_Connect, q, Num_Link):    #Given the probability of connections between two nodes and q the probabilit of 
                                                  #links being removed the total number of links being removed is computed as follows
    L=0
    for i in range(0, Num_Link):
        L= L + (q * Pr_Connect[i])        
    return L


#This function compute the maximum achievable precision based on our paper https://doi.org/10.1103/PhysRevE.101.052318      
def Expected_Precision_OS (Pr_values, q, L) :
    H=0
    T=0  
    Telorance=1
    
    itr=0
    while abs(L-H) > Telorance:
        T= T + (q * Pr_values[itr])
        H= H + 1- Pr_values[itr] + (q * Pr_values[itr])
        itr= itr+1
    Res = T/H 
    return Res
            

if __name__ == '__main__':
    config = load(open(sys.argv[1], "r"), Loader=Loader)

    for parameter in config:
        if 'Beta_s' in parameter.keys():
            Beta_s = parameter['Beta_s']
        elif 'gamma_s' in parameter.keys():
            gamma_s = parameter['gamma_s']
        elif 'Ns_obs' in parameter.keys():
            Ns_obs = parameter['Ns_obs']
        elif 'kmean_s' in parameter.keys():
            kmean_s = parameter['kmean_s']
        elif 'gamma_n' in parameter.keys():
            gamma_n = parameter['gamma_n']
        elif 'kmean_n' in parameter.keys():
            kmean_n = parameter['kmean_n']
        elif 'gamma_f' in parameter.keys():
            gamma_f = parameter['gamma_f']
        elif 'N_f' in parameter.keys():
            N_f = parameter['N_f']
        elif 'Beta_bi' in parameter.keys():
            Beta_bi = parameter['Beta_bi']
        elif 'nu' in parameter.keys():
            nu = parameter['nu']
        elif 'alpha' in parameter.keys():
            alpha = parameter['alpha']
        elif 'N_labels' in parameter.keys():
            N_labels = parameter['N_labels']
        elif 'ntimes' in parameter.keys():
            ntimes = parameter['ntimes']

all_params = [Beta_s, gamma_s, Ns_obs, kmean_s, gamma_n, kmean_n, gamma_f, N_f, Beta_bi, nu, alpha, N_labels]
Q= np.arange(0, 1.05, 0.05)
Q[0]=0.01

for (Beta_s_val, gamma_s_val, Ns_obs_val, kmean_s_val, gamma_n_val, kmean_n_val, gamma_f_val, N_f_val, Beta_bi_val, nu_val, alpha_val, N_labels_val) in itertools.product(*all_params):
    for i in range(ntimes):
        Folder_name=f'output_B_s_{Beta_s_val}_g_s_{gamma_s_val}_Ns_obs_{Ns_obs_val}_k_s_{kmean_s_val}_g_n_{gamma_n_val}_k_n_{kmean_n_val}_g_f_{gamma_f_val}_N_f_{N_f_val}_B_bi_{Beta_bi_val}_nu_{nu_val}_alpha_{alpha_val}_N_l_{N_labels_val}_i_{i}'
        print(Folder_name)
        log_file= Folder_name + "/log.txt"
        # open the text file and read its contents
        with open(log_file, 'r') as file:
            contents = file.read()
        
        # search for the "Final mu" value in the file contents
        pattern = r"Final mu = ([-]?[0-9]+[,.]?[0-9]*([\/][0-9]+[,.]?[0-9]*)*)"
        match = re.search(pattern, contents)
        
        if match:
            # extract the numeric value of "Final mu"
            mu = match.group(1)
            print("Final mu:", mu)
        else:
            print("Final mu not found")
         
        
        Coord_file= f'{Folder_name}/Net_N_{Ns_obs_val}_g_{gamma_s_val:.2f}_B_{Beta_s_val:.2f}_k_{kmean_s_val:.2f}_a_{alpha_val:.2f}_nc_{N_labels_val}.unipartite.coordinates'        
        print(Coord_file)
        

        df= pd.read_csv(Coord_file, delimiter='\t', header=None)

        Kappas=np.array(df.iloc[:][1])
        Thetas=np.array(df.iloc[:][2])
        Net_size= max(df.iloc[:][0])+1  #We assume that node labels start from 0
        
    

        Link_size= int (Net_size * (Net_size-1)/2)  
        Pr_Connection= np.zeros(Link_size , dtype=float)
        R= Net_size/(2 * math.pi)
        Ind=0
        for i in range(0, Net_size-1):
            for j in range(i+1, Net_size):
                Delta_Theta = math.pi - abs(math.pi - abs(Thetas[i] - Thetas[j]))
                Dist= R * Delta_Theta
                Prob= 1/ (1 + pow( (Dist/(float(mu) * Kappas[i] * Kappas[j])) , Beta_s_val))
                Pr_Connection[Ind]= Prob
                Ind= Ind+1  
                
        with open(f'{Folder_name}/Max_Prc.txt', "w") as file:
            Pr_values = np.sort(Pr_Connection, order=None)[::-1]
            for q in Q:  
                print("q = {:.2f}".format(q))
                L= Num_Removed_link(Pr_Connection, q, Link_size)
                Pr= Expected_Precision_OS (Pr_values, q, L)
                file.write(f'{q:.2f}' + "\t" + f'{Pr}' + "\n")
                
            