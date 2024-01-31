import numpy as np
import pandas as pd
import argparse
import textwrap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import copy
from tqdm import tqdm
import os
from numba import njit


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent("""
    
    Compute the maximum accuracy of node classification for S1 synthetic network.

    """))

    parser.add_argument('-c', '--coords', type=str,
                        required=True, help="Path to coordinates file")

    parser.add_argument('--test_size_min', type=float, required=False,
                        default=0.05, help="Minimum size of the test set")
    parser.add_argument('--test_size_max', type=float, required=False,
                        default=0.95, help="Maximum size of the test set")
    parser.add_argument('--test_size_num', type=int, required=False,
                        default=20, help="Number of samples of test size")

    args = parser.parse_args()
    return args

@njit
def angle(t1, t2):
    return np.pi - np.fabs(np.pi - np.fabs(t1 - t2))


def compute_label_probability(node_theta, cluster_num, cluster_thetas, alpha):
    dij = angle(node_theta, cluster_thetas[cluster_num])
    norm = np.sum([np.power(angle(node_theta, d), -alpha)
                  for d in cluster_thetas])
    return np.power(dij, -alpha) / norm


def read_coordinates(path):
    df = pd.read_csv(path, sep="\t", header=None)
    df.columns = ['index', 'kappa', 'theta', 'label']
    return df


def assign_labels(thetas, centers, alpha):
    labels = []
    for t in thetas:
        total_distance = sum([np.power(angle(t, c), -alpha) for c in centers]) 
        prob = [np.power(angle(t, c), -alpha) / total_distance for c in centers]
        l = np.random.choice(len(prob), size=1, p=prob)[0]
        labels.append(l)
    return labels


def compute_accuracy(thetas, labels, cluster_positions, alpha):
    predicted_labels = assign_labels(thetas, cluster_positions, alpha)
    return accuracy_score(labels, predicted_labels)


def compute_log_f(alpha, thetas_center, thetas_train, labels_train, precomputed_scaling):
    total_prob = 0
    for i in range(len(thetas_train)):
        left = -alpha * np.log(angle(thetas_train[i], thetas_center[labels_train[i]]))
        right = -np.log(sum(x[i] for x in precomputed_scaling))
        total_prob += left + right
    return total_prob


@njit
def compute_scaling_for_center(alpha, center, thetas_train):
    scaling = []
    for t in thetas_train:
        scaling.append(np.power(angle(center, t), -alpha))
    return scaling


def infer_parameters_per_alpha(alpha, thetas_train, labels_train, num_clusters, niter=5000, T=0.5):
    centers = np.random.uniform(0, 2*np.pi, size=num_clusters)
    tmp_centers = copy.deepcopy(centers)

    all_scaling = []
    for c in centers:
        scaling = compute_scaling_for_center(alpha, c, thetas_train)
        all_scaling.append(scaling)

    tmp_all_scaling = copy.deepcopy(all_scaling)

    current_loglikelihood = compute_log_f(alpha, centers, thetas_train, labels_train, all_scaling)

    for _ in range(niter):
        # Choose random center to move
        c = np.random.choice(num_clusters, size=1)[0]
        new_position = np.random.uniform(0, 2*np.pi)
        
        tmp_all_scaling[c] = compute_scaling_for_center(alpha, new_position, thetas_train)
        tmp_centers[c] = new_position

        new_loglikelihood = compute_log_f(alpha, tmp_centers, thetas_train, labels_train, tmp_all_scaling)
            
        prob = np.exp((new_loglikelihood - current_loglikelihood) / T)
        if np.random.rand() < prob: # accept the new position
            centers[c] = new_position
            all_scaling[c] = tmp_all_scaling[c]
            current_loglikelihood = new_loglikelihood
        else:
            tmp_all_scaling[c] = all_scaling[c]
            tmp_centers[c] = centers[c]
    
    return centers, current_loglikelihood



def infer_parameters(thetas_train, labels_train, num_clusters):
    alphas = np.arange(-10, 11)
    highest_likelihood = -1e10
    inferred_centers = []
    inferred_alpha = 0

    all_likelihood = []

    for a in tqdm(alphas):
        centers, likelihood = infer_parameters_per_alpha(a, thetas_train, labels_train, num_clusters)
        all_likelihood.append(likelihood)
        if likelihood > highest_likelihood:
            inferred_centers = centers
            inferred_alpha = a
            highest_likelihood = likelihood
    
    return [inferred_alpha, *inferred_centers], [alphas, all_likelihood]


if __name__ == '__main__':
    args = parse_args()

    df = read_coordinates(args.coords)
    num_clusters = len(np.unique(df['label']))

    test_sizes = np.linspace(
        args.test_size_min, args.test_size_max, num=args.test_size_num)
    max_predicted_train_accuracies = []
    max_predicted_test_accuracies = []
    all_final_parameters = []
    all_estimated_alphas = []
    all_estimated_max_likelihood = []

    for t in tqdm(test_sizes):
        thetas_train, thetas_test, labels_train, labels_test = train_test_split(
            df['theta'].values, df['label'].values, test_size=t)

        final_parameters, estimated_parameters = infer_parameters(
            thetas_train, labels_train, num_clusters)
        print(final_parameters)
        print(estimated_parameters)

        all_estimated_alphas.append(estimated_parameters[0])
        all_estimated_max_likelihood.append(estimated_parameters[1])

        max_predicted_train_accuracy = compute_accuracy(
            thetas_train, labels_train, final_parameters[1:], final_parameters[0])
        max_predicted_test_accuracy = compute_accuracy(
            thetas_test, labels_test, final_parameters[1:], final_parameters[0])

        max_predicted_train_accuracies.append(max_predicted_train_accuracy)
        max_predicted_test_accuracies.append(max_predicted_test_accuracy)

        all_final_parameters.append(final_parameters)


    result_df = pd.DataFrame()
    result_df['test_size'] = test_sizes
    result_df['max_predicted_train_accuracy'] = max_predicted_train_accuracies
    result_df['max_predicted_test_accuracy'] = max_predicted_test_accuracies
    parameters_columns = ['alpha', *[f'theta_{i}' for i in range(num_clusters)]]
    
    n_alphas = len(all_estimated_alphas[0])
    alpha_estimated_columns = [f'alpha_estimated_{i}' for i in range(n_alphas)]
    result_df[alpha_estimated_columns] = all_estimated_alphas
    likelihood_estimated_columns = [f'likelihood_estimated_{i}' for i in range(n_alphas)]
    result_df[likelihood_estimated_columns] = all_estimated_max_likelihood
    
    result_df[parameters_columns] = all_final_parameters
    result_df.to_csv(f'{os.path.dirname(args.coords)}/max_accuracy.csv', index=False)
    print(result_df)