import numpy as np
import pandas as pd
import argparse
import textwrap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.optimize import dual_annealing, minimize
from tqdm import tqdm
import os


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent("""
    
    Compute the maximum accuracy of node classification for S1 synthetic network.

    """))

    parser.add_argument('-c', '--coords', type=str,
                        required=True, help="Path to coordinates file")

    parser.add_argument('--test_size_min', type=float, required=False,
                        default=0.01, help="Minimum size of the test set")
    parser.add_argument('--test_size_max', type=float, required=False,
                        default=0.99, help="Maximum size of the test set")
    parser.add_argument('--test_size_num', type=int, required=False,
                        default=20, help="Number of samples of test size")

    args = parser.parse_args()
    return args


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


def compute_accuracy(thetas, labels, cluster_positions, alpha):
    num_clusters = len(cluster_positions)
    probs = []
    for n in range(num_clusters):
        probs_per_cluster = [compute_label_probability(
            t, n, cluster_positions, alpha) for t in thetas]
        probs.append(probs_per_cluster)

    probs = np.array(probs)
    predicted_labels = np.argmax(probs.T, axis=1)
    return accuracy_score(labels, predicted_labels)


def f(parameters, nodes_thetas, nodes_labels):
    alpha, *cluster_thetas = parameters
    total_prob = 1
    for i in range(len(nodes_thetas)):
        top = np.power(
            angle(nodes_thetas[i], cluster_thetas[nodes_labels[i]]), -alpha)
        bottom = np.sum([np.power(angle(nodes_thetas[i], d), -alpha)
                        for d in cluster_thetas])
        prob = top / bottom
        total_prob *= prob
    #return 1 - total_prob
    return -np.log(total_prob + 1e-10)


def log_f(parameters, nodes_thetas, nodes_labels):
    alpha, *cluster_thetas = parameters
    total_prob = 0
    for i in range(len(nodes_thetas)):
        left = -alpha * np.log(angle(nodes_thetas[i], cluster_thetas[nodes_labels[i]]))
        right = -np.log(np.sum([np.power(angle(nodes_thetas[i], d), -alpha)
                        for d in cluster_thetas]))
        log_prob = left + right
        total_prob += log_prob
    return -total_prob

def minimize_accuracy(parameters, nodes_thetas, nodes_labels):
    alpha, *cluster_thetas = parameters
    acc = compute_accuracy(
            nodes_thetas, nodes_labels, cluster_thetas, alpha)
    return -acc


def infer_parameters(thetas_train, labels_train, num_clusters, n_init=20):
    bounds = ((-20, 20), *tuple([(0, 2*np.pi) for _ in range(num_clusters)]))
    final_parameters = []
    current_acc = 0

    for _ in tqdm(range(n_init)):
        initial_alpha = np.random.uniform(bounds[0][0], bounds[0][1])
        initial_angles = np.random.uniform(0, 2*np.pi, size=num_clusters)

        res = minimize(log_f, [initial_alpha, *initial_angles],
                       args=(thetas_train, labels_train), bounds=bounds)
        predicted_train_accuracy = compute_accuracy(
            thetas_train, labels_train, res.x[1:], res.x[0])
        
        if predicted_train_accuracy > current_acc:
            current_acc = predicted_train_accuracy
            final_parameters = res.x

    return final_parameters


def global_infer_parameters(thetas_train, labels_train, num_clusters):
    bounds = ((-20, 20), *tuple([(0, 2*np.pi) for _ in range(num_clusters)]))
    res = dual_annealing(log_f, args=(thetas_train, labels_train), bounds=bounds)
    print(res)
    return res.x


if __name__ == '__main__':
    args = parse_args()

    df = read_coordinates(args.coords)
    num_clusters = len(np.unique(df['label']))

    test_sizes = np.linspace(
        args.test_size_min, args.test_size_max, num=args.test_size_num)
    max_predicted_train_accuracies = []
    max_predicted_test_accuracies = []
    all_final_parameters = []

    for t in tqdm(test_sizes):
        thetas_train, thetas_test, labels_train, labels_test = train_test_split(
            df['theta'].values, df['label'].values, test_size=t)

        final_parameters = infer_parameters(
            thetas_train, labels_train, num_clusters)

        max_predicted_train_accuracy = compute_accuracy(
            thetas_train, labels_train, final_parameters[1:], final_parameters[0])
        max_predicted_test_accuracy = compute_accuracy(
            thetas_test, labels_test, final_parameters[1:], final_parameters[0])

        max_predicted_train_accuracies.append(max_predicted_train_accuracy)
        max_predicted_test_accuracies.append(max_predicted_test_accuracy)

        print(final_parameters)
        all_final_parameters.append(final_parameters)

    result_df = pd.DataFrame()
    result_df['test_size'] = test_sizes
    result_df['max_predicted_train_accuracy'] = max_predicted_train_accuracies
    result_df['max_predicted_test_accuracy'] = max_predicted_test_accuracies
    parameters_columns = ['alpha', *[f'theta_{i}' for i in range(num_clusters)]]
    result_df[parameters_columns] = all_final_parameters
    result_df.to_csv(f'{os.path.dirname(args.coords)}/max_accuracy.csv', index=False)
    print(result_df)