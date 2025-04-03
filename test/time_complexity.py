import time
import os
from tqdm import tqdm

if __name__ == '__main__':
    sizes = [500, 1000, 2000, 5000, 10000]
    beta = beta_b = 3
    kmean = kmean_n = 10
    gamma = gamma_n = gamma_f = 2.7
    threads = [1, 2, 4]

    total_iterations = len(sizes) * len(threads) * 50

    with open('results/time_complexity_network_generation_new_24_03_25_new3.txt', 'a') as file:
        file.write('size,thread,i,time\n')
        # Create a tqdm progress bar for the total number of iterations
        with tqdm(total=total_iterations, desc="Overall Progress") as pbar:
            for N in sizes:
                for t in threads:
                    for i in range(50):
                        # Update the progress bar description with current parameters
                        pbar.set_description(f"Size: {N}, Threads: {t}, Iteration: {i}")
                        start = time.time()
                        command = (
                            f'./src/main_run_parallel {beta} {gamma} {N} {kmean} '
                            f'{gamma_n} {kmean_n} {gamma_f} {N} {beta_b} {0} {10} {5} '
                            f'/tmp/output_nets/ {t} > log.txt'
                        )
                        os.system(command)
                        elapsed_time = time.time() - start
                        file.write(f'{N},{t},{i},{elapsed_time}\n')
                        pbar.update(1)