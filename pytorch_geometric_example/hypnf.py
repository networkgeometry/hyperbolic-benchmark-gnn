import torch
from torch_geometric.data import Data
from torch_geometric.datasets.graph_generator import GraphGenerator
import random
import numpy as np


class HypNF(GraphGenerator):
    r"""Generate HypNF model

    Args:
        N_n (int): Number of nodes.
        beta (float): Inverse temperature, controlling the clustering coefficient.
                      Must be greater than 1.
        gamma (float): Exponent of the power-law distribution for hidden degrees
                       `kappa` for unipartite network.
        kmean (float): The mean degree of the unipartite network.

        N_f (int): Number of features.
        beta_b (float): Controls bipartite clustering. Must be greater than 1.
        gamma_n (float): Exponent of the power-law distribution for hidden degrees
                         `kappa` of nodes in the bipartite network.
        gamma_f (float): Exponent of the power-law distribution for hidden degrees
                         `kappa` of features in the bipartite network.
        kmean_n (float):  The mean degree nodes in the bipartite network.

        N_c (int): Number of classess.
        alpha (float): Tunes the level of homophily in the network.
    """

    def __init__(
        self,
        N_n: int,
        beta: float,
        gamma: float,
        kmean: float,
        N_f: int,
        beta_b: float,
        gamma_n: float,
        gamma_f: float,
        kmean_n: float,
        N_c: int,
        alpha: float,
    ):
        super().__init__()
        self.N_n = N_n
        self.beta = beta
        self.gamma = gamma
        self.kmean = kmean
        self.N_f = N_f
        self.beta_b = beta_b
        self.gamma_n = gamma_n
        self.gamma_f = gamma_f
        self.kmean_n = kmean_n
        self.N_c = N_c
        self.alpha = alpha
        self.R = self.N_n / (2 * np.pi)

        # Generate hidden degrees
        self.kappas = _generate_power_law_distribution(N_n, gamma, kmean)
        self.kappas_n = _generate_power_law_distribution(N_n, gamma_n, kmean_n)
        kmean_f = N_n / N_f * kmean_n
        self.kappas_f = _generate_power_law_distribution(N_f, gamma_f, kmean_f)

        # Generate angular positions
        self.thetas = np.random.uniform(0, 2 * np.pi, N_n)
        self.thetas_f = np.random.uniform(0, 2 * np.pi, N_f)

        # Compute parameters mu and mu_b
        self.mu = beta / (2 * np.pi * kmean) * np.sin(np.pi / beta)
        self.mu_b = beta_b / (2 * np.pi * kmean_n) * np.sin(np.pi / beta_b)

        # Compute radial coordinates
        kappa_min = np.min(self.kappas)
        Rhat = 2 * np.log(2 * self.R / (self.mu * kappa_min**2))
        self.radii = [Rhat - 2 * np.log(kappa / kappa_min) for kappa in self.kappas]
        kappa_n_min = np.min(self.kappas_n)
        kappa_f_min = np.min(self.kappas_f)
        Rhat_b = 2 * np.log(2 * self.R / (self.mu_b * kappa_n_min * kappa_f_min))
        self.radii_n = [
            Rhat_b - 2 * np.log(kappa_n / kappa_n_min) for kappa_n in self.kappas_n
        ]
        self.radii_f = [
            Rhat_b - 2 * np.log(kappa_f / kappa_f_min) for kappa_f in self.kappas_f
        ]

    def __call__(self) -> Data:
        # Generate unipartite network
        source_nodes = []
        target_nodes = []
        for u in range(self.N_n):
            for v in range(u):
                angle = _get_angle(self.thetas[u], self.thetas[v])
                chi = self.R * angle / (self.mu * self.kappas[u] * self.kappas[v])
                p_ij = 1 / (1 + np.power(chi, self.beta))
                if random.random() < p_ij:
                    source_nodes.append(u)
                    target_nodes.append(v)
                    source_nodes.append(v)
                    target_nodes.append(u)
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

        # Generate bipartite network
        x = torch.zeros((self.N_n, self.N_f), dtype=torch.float)
        for n in range(self.N_n):
            for f in range(self.N_f):
                angle = _get_angle(self.thetas[n], self.thetas_f[f])
                chi = self.R * angle / (self.mu_b * self.kappas_n[n] * self.kappas_f[f])
                p_ij = 1 / (1 + np.power(chi, self.beta_b))
                if random.random() < p_ij:
                    x[n, f] = 1

        y = _generate_labels(self.N_c, self.alpha, self.thetas)

        return Data(
            x=x,
            edge_index=edge_index,
            thetas=self.thetas,
            kappas=self.kappas,
            radii=self.radii,
            thetas_f=self.thetas_f,
            kappas_f=self.kappas_f,
            kappas_n=self.kappas_n,
            radii_n=self.radii_n,
            radii_f=self.radii_f,
            y=torch.tensor(y),
            num_nodes=self.N_n,
            num_node_features=self.N_f,
            num_classes=self.N_c,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"N_n={self.N_n}, "
            f"beta={self.beta}, "
            f"gamma={self.gamma}, "
            f"kmean={self.kmean}, "
            f"N_f={self.N_f}, "
            f"beta_b={self.beta_b}, "
            f"gamma_n={self.gamma_n}, "
            f"gamma_f={self.gamma_f}, "
            f"kmean_n={self.kmean_n}, "
            f"N_c={self.N_c}, "
            f"alpha={self.alpha}"
            f")"
        )


def _generate_power_law_distribution(n: int, gamma: float, kmean: float):
    gam_ratio = (gamma - 2) / (gamma - 1)
    kappa_0 = kmean * gam_ratio * (1 - 1 / n) / (1 - 1 / n**gam_ratio)
    base = 1 - 1 / n
    power = 1 / (1 - gamma)
    kappas = [kappa_0 * (1 - random.random() * base) ** power for _ in range(n)]
    return kappas


def _generate_labels(N_c, alpha, thetas):
    centers = np.random.uniform(0, 2 * np.pi, size=N_c)
    labels = []
    for t in thetas:
        total_distance = sum([np.power(_get_angle(t, c), -alpha) for c in centers])
        prob = [np.power(_get_angle(t, c), -alpha) / total_distance for c in centers]
        l = np.random.choice(len(prob), size=1, p=prob)[0]
        labels.append(l)
    return labels


def _get_angle(t1, t2):
    return np.pi - np.fabs(np.pi - np.fabs(t1 - t2))
