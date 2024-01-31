#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 09:29:47 2023

@author: complexity-lab1

This code generates networks using the S1*S1 model with a fixed value of Beta_s set to 1.2.
It proceeds to extract the unipartite network and applies randomization with a new value of New_Beta_s that is less than one.
In the Tree.yaml file, the range of New_Beta_s values is specified as [0.5, 0.9].

Afterwards, the code renames the folders and files by replacing Beta_s with New_Beta_s. 
Additionally, it retains the original edge list corresponding to Beta_s = 1.2 by appending "_init" to its name.

To be run with : python3 generate_Tree_like_unipartite_Nets.py Tree.yaml 

"""

import os
import sys
from yaml import load, Loader
import itertools
import glob
import time
import re
import shutil
import asyncio
import numpy as np


def background(f):
    # From https://stackoverflow.com/a/59385935
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    return wrapped

def construct_folder(Beta_s, gamma_s, Ns_obs, kmean_s, gamma_n, kmean_n, gamma_f, N_f, Beta_bi, nu, alpha, N_labels, i):
    if int(alpha) < 0:
        alpha = f'neg{-int(alpha)}'
    return f'output_B_s_{Beta_s}_g_s_{gamma_s}_Ns_obs_{Ns_obs}_k_s_{kmean_s}_g_n_{gamma_n}_k_n_{kmean_n}_g_f_{gamma_f}_N_f_{N_f}_B_bi_{Beta_bi}_nu_{nu}_alpha_{alpha}_N_l_{N_labels}_i_{i}/'
 
        
def rename_directory(old_name, new_name):
    try:
        os.makedirs(new_name)  # Create the new directory
        for item in os.listdir(old_name):  # Iterate over items in the original directory
            item_path = os.path.join(old_name, item)
            shutil.move(item_path, new_name)  # Move each item to the new directory
        os.rmdir(old_name)  # Remove the original directory
        print(f"Directory '{old_name}' renamed to '{new_name}' successfully.")
    except FileNotFoundError:
        print(f"Directory '{old_name}' does not exist.")
    except FileExistsError:
        print(f"A directory with the name '{new_name}' already exists.")
   
@background
def generate_networks(ntimes, Beta_s_val, gamma_s_val, Ns_obs_val, kmean_s_val, gamma_n_val, kmean_n_val, gamma_f_val, N_f_val, Beta_bi_val, nu_val, alpha_val, N_labels_val, New_Beta_s_val):
    for i in range(ntimes):
        log_filename = f'log_{np.random.randint(0, 1000000)}.txt'
        output_folder = construct_folder(
                        Beta_s_val, gamma_s_val, Ns_obs_val, kmean_s_val, gamma_n_val, kmean_n_val, gamma_f_val, N_f_val, Beta_bi_val, nu_val, alpha_val, N_labels_val, i)
        command = f'./main_robert3 {Beta_s_val} {gamma_s_val} {Ns_obs_val} {kmean_s_val} {gamma_n_val} {kmean_n_val} {gamma_f_val} {N_f_val} {Beta_bi_val} {nu_val} {alpha_val} {N_labels_val} {output_folder} > {log_filename}'
        print(command)
        os.system(command)
        time.sleep(1)

        # Extract labels from coordinate file
        coordiantes = glob.glob(f'{output_folder}/*.unipartite.coordinates')[0]
        label_path = coordiantes.replace('coordinates', 'labels')
        extract_labels_command = f'cat {coordiantes} | cut -f4 > {label_path}'
        os.system(extract_labels_command)

        # Move the log file
        os.system(f'mv {log_filename} {output_folder}/{log_filename}')                
        #~~~~~~~~~~~~~~~~#Rename the folder using the new value for Beta (smaller than one )
        if int(alpha_val) < 0:
            alpha_val = f'neg{-int(alpha_val)}'

        new_output_folder= f'output_B_s_{New_Beta_s_val}_g_s_{gamma_s_val}_Ns_obs_{Ns_obs_val}_k_s_{kmean_s_val}_g_n_{gamma_n_val}_k_n_{kmean_n_val}_g_f_{gamma_f_val}_N_f_{N_f_val}_B_bi_{Beta_bi_val}_nu_{nu_val}_alpha_{alpha_val}_N_l_{N_labels_val}_i_{i}' 
        rename_directory(output_folder, new_output_folder)
        
        #~~~~~~~~~~~~~~~~~~~~~~Run Geometric randomization on the unipartite network                
        Unipartite_Net = glob.glob(f'{new_output_folder}/*.unipartite.edgelist')[0]                               
        print(Unipartite_Net)
        
        Coord_file = glob.glob(f'{new_output_folder}/*.unipartite.coordinates')[0]
        print(Coord_file)
        
        log_file= f"{new_output_folder}/{log_filename}"
        # open the text file and read its contents
        with open(log_file, 'r') as file:
            contents = file.read()
        
        pattern = r"Final mu = ([-]?[0-9]+[,.]?[0-9]*([\/][0-9]+[,.]?[0-9]*)*)"
        match = re.search(pattern, contents)
        
        if match:
            # extract the numeric value of "Final mu"
            mu = match.group(1)
            print("Final mu:", mu)
        else:
            print("Final mu not found")
            
        log_GR_filename = f'log_GR_{np.random.randint(0, 1000000)}.txt'
        command = f'./main_Tree  {Unipartite_Net} {Coord_file} {mu} {Beta_s_val} {New_Beta_s_val} > {log_GR_filename}'
        print(command)
        os.system(command)               
        os.system(f'mv {log_GR_filename} {new_output_folder}/{log_GR_filename}')
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Rename some files            
                        
        Unipartite_Net_new= Unipartite_Net + "_init"
        os.rename(Unipartite_Net, Unipartite_Net_new)   #keep the initial edgelist by adding _init to its name

        #Rename the output of the C file (The edgelist of the randomized network using beta smaller than one)               
        GR_old_file= glob.glob(f'{new_output_folder}/*.rand')[0]                                                        
        GR_new_file=Unipartite_Net.replace(f'B_{Beta_s_val:.2f}', f'B_{New_Beta_s_val:.2f}')    
        os.rename(GR_old_file, GR_new_file)      
        
        
        Coord_old_file= Coord_file
        Coord_new_file= Coord_old_file.replace (f'B_{Beta_s_val:.2f}', f'B_{New_Beta_s_val:.2f}')
        print(Coord_new_file)
        print(Coord_old_file)
        os.rename(Coord_old_file, Coord_new_file)   #rename coordinate file 
        
        label_old_file= glob.glob(f'{new_output_folder}/*.labels')[0]
        label_new_file= label_old_file.replace(f'B_{Beta_s_val:.2f}', f'B_{New_Beta_s_val:.2f}')
        print(label_new_file)
        print(label_old_file)
        os.rename(label_old_file, label_new_file) #rename label file    


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
        elif 'New_Beta_s' in parameter.keys():
            New_Beta_s= parameter['New_Beta_s']

    os.system('gcc -O3 -o main_robert3 main_robert3.c -lm')    
    os.system('gcc -O3 -o main_Tree  Tree_like_Networks.c -lm')
    
    all_params = [Beta_s, gamma_s, Ns_obs, kmean_s, gamma_n, kmean_n, gamma_f, N_f, Beta_bi, nu, alpha, N_labels]

    loop = asyncio.get_event_loop()
    
    for New_Beta_s_val in New_Beta_s:
        looper = asyncio.gather(*[
            generate_networks(ntimes, Beta_s_val, gamma_s_val, Ns_obs_val, kmean_s_val, gamma_n_val, kmean_n_val, gamma_f_val, N_f_val, Beta_bi_val, nu_val, alpha_val, N_labels_val, New_Beta_s_val) 
            for (Beta_s_val, gamma_s_val, Ns_obs_val, kmean_s_val, gamma_n_val, kmean_n_val, gamma_f_val, N_f_val, Beta_bi_val, nu_val, alpha_val, N_labels_val) in itertools.product(*all_params)])
        results = loop.run_until_complete(looper)

                
                
              
