import os
import sys
from yaml import load, Loader
import itertools
import glob
import time

def construct_folder(Beta_s, gamma_s, Ns_obs, kmean_s, gamma_n, kmean_n, gamma_f, N_f, Beta_bi, nu, alpha, N_labels, i):
    if int(alpha) < 0:
        alpha = f'neg{-int(alpha)}'
    return f'output_B_s_{Beta_s}_g_s_{gamma_s}_Ns_obs_{Ns_obs}_k_s_{kmean_s}_g_n_{gamma_n}_k_n_{kmean_n}_g_f_{gamma_f}_N_f_{N_f}_B_bi_{Beta_bi}_nu_{nu}_alpha_{alpha}_N_l_{N_labels}_i_{i}/'


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

    os.system('gcc -O3 -o main_robert2 main_robert2.c -lm')

    all_params = [Beta_s, gamma_s, Ns_obs, kmean_s, gamma_n, kmean_n, gamma_f, N_f, Beta_bi, nu, alpha, N_labels]
    for i in range(ntimes):
        for (Beta_s_val, gamma_s_val, Ns_obs_val, kmean_s_val, gamma_n_val, kmean_n_val, gamma_f_val, N_f_val, Beta_bi_val, nu_val, alpha_val, N_labels_val) in itertools.product(*all_params):
            output_folder = construct_folder(
                Beta_s_val, gamma_s_val, Ns_obs_val, kmean_s_val, gamma_n_val, kmean_n_val, gamma_f_val, N_f_val, Beta_bi_val, nu_val, alpha_val, N_labels_val, i)
            command = f'./main_robert2 {Beta_s_val} {gamma_s_val} {Ns_obs_val} {kmean_s_val} {gamma_n_val} {kmean_n_val} {gamma_f_val} {N_f_val} {Beta_bi_val} {nu_val} {alpha_val} {N_labels_val} {output_folder} > log.txt'
            print(command)
            os.system(command)
            time.sleep(1)

            # Extract labels from coordinate file
            coordiantes = glob.glob(f'{output_folder}/*.unipartite.coordinates')[0]
            label_path = coordiantes.replace('coordinates', 'labels')
            extract_labels_command = f'cat {coordiantes} | cut -f4 > {label_path}'
            os.system(extract_labels_command)

            # Move the log file
            os.system(f'mv log.txt {output_folder}/log.txt')