import numpy as np


def rand_index(clusters, groups):
    """
    shape (clusters): n_samples
    shape (groups): n_samples
    """
    TP_total = 0
    TN_total = 0
    FP_total = 0
    FN_total = 0

    for l in range(len(clusters)):
        SC = clusters[l] == clusters[(l + 1):]
        SG = groups[l] == groups[(l + 1):]

        TP = np.logical_and(SG, SC)
        FP = np.logical_and(np.logical_not(SG), SC)
        FN = np.logical_and(SG, np.logical_not(SC))
        TN = np.logical_and(np.logical_not(SG), np.logical_not(SC))

        TP_total += np.sum(TP)
        TN_total += np.sum(TN)
        FP_total += np.sum(FP)
        FN_total += np.sum(FN)

    return np.array([[TP_total, FP_total],
                     [FN_total, TN_total]])


def randMetrics(rand_index, N):
    precision = rand_index[0, 0] / (rand_index[0, 0] + rand_index[0, 1])
    recall = rand_index[0, 0] / (rand_index[0, 0] + rand_index[1, 0])
    rand = (rand_index[0, 0] + rand_index[1, 1]) / ((N * (N - 1)) / 2)
    F1 = 2 * ((precision * recall) / (precision + recall))

    print("Precision: {precision}\nRecall: {recall}\nRand: {rand}\nF1: {F1}".format(precision=precision, recall=recall,
                                                                                    rand=rand, F1=F1))
