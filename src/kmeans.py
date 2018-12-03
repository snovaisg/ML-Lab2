from utils.utils import load_data
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

df = load_data()

clusters = KMeans(n_clusters=2, random_state=0).fit_predict(df[["x", "y", "z"]])
print("silhouette score by faults: ", silhouette_score(df[["x", "y", "z"]], clusters))


def gen_sil_score_by_cluster(n_clusters, file_path):
    """
    saves a file with n_clusters-1 clusters built using kmeans and computes the silhouette score for
    each one.

    :param n_clusters: max size of clusters that we want to test kmeans
    :param file_path: path to save a csv with the size of clusters and their silhouette scores
    :return: Bool
    """

    def study_silhouette_by_cluster(size_clusters):
        """
        Generates size_clusters -1 clusters with kmeans and returns their silhouette score

        :param size_clusters: max size of clusters that we want to test kmeans
        :return: numpy ndarray, shape = (size_clusters-1,2) with (cluster_size,silhouette_score)
        """
        mem = np.ndarray(shape=(n_clusters - 1, 2))
        for size_cluster in range(2, size_clusters + 1):
            clusters = KMeans(n_clusters=size_cluster, random_state=0).fit_predict(df[["x", "y", "z"]])
            score = silhouette_score(df[["x", "y", "z"]], clusters)
            mem[size_cluster - 2] = np.array([size_cluster, score])
            mem = mem[mem[:, 1].argsort()][::-1]
        return mem

    sil_results = pd.DataFrame(study_silhouette_by_cluster(n_clusters),
                               columns=["size_cluster", "sil_score"]).set_index("size_cluster")
    sil_results.to_csv(file_path)
    return True


gen_sil_score_by_cluster(40, "../save/kmeans/sil_score.csv")
