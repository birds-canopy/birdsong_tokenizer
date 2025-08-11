import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.special import gamma

def compute_hypersphere_volume(radius, dim):
    """Volume d'une hypersphère de dimension dim et de rayon radius."""
    return (np.pi ** (dim / 2) * radius ** dim) / gamma(dim / 2 + 1)

def mean_cluster_hypersphere_density(X, k):
    """
    Calcule la densité moyenne des clusters en supposant chaque cluster
    comme une hypersphère (comme décrit en III.B du papier).
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    dim = X.shape[1]

    densities = []
    for i in range(k):
        cluster_pts = X[labels == i]
        if len(cluster_pts) == 0:
            continue
        # Rayon = distance maximale d'un point au centroïde
        distances = np.linalg.norm(cluster_pts - centers[i], axis=1)
        radius = np.max(distances)
        volume = compute_hypersphere_volume(radius, dim)
        density = volume / len(cluster_pts) if len(cluster_pts) > 0 else 0 # Cf article, je trouvais ça bizarre mais c ok
        densities.append(density)

    if not densities:
        return 0
    return np.mean(densities)

def detect_elbow_k(k_list, density_list):
    """
    Détection automatique du coude via la plus grande distance à la
    ligne reliant le premier et le dernier point (méthode du "knee").
    """
    # Normalisation pour stabilité
    ks = np.array(k_list, dtype=float)
    dens = np.array(density_list, dtype=float)
    plt.plot(ks, dens, marker='o')
    plt.title('Hypersphere Density vs number of Clusters')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Hypersphere Density')
    # Ligne entre (k0, d0) et (kN, dN)
    p1 = np.array([ks[0], dens[0]])
    p2 = np.array([ks[-1], dens[-1]])
    # Calcul de la distance de chaque point à la ligne
    line_vec = p2 - p1
    point_vecs = np.stack([ks, dens], axis=1) - p1  # shape (n,2)
    # Projection scalaires
    proj = (point_vecs @ line_vec) / (np.dot(line_vec, line_vec))
    proj_points = np.outer(proj, line_vec) + p1  # projection sur la ligne
    # Vecteurs perpendiculaires
    diff = np.stack([ks, dens], axis=1) - proj_points
    distances = np.linalg.norm(diff, axis=1)
    elbow_idx = np.argmax(distances)
    return int(ks[elbow_idx])

