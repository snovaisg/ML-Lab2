import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imageio import imread





def load_data(columns=["id", "latitude", "longitude", "fault", "time", "type"]):
    df = pd.read_csv("../data/tp2_data.csv")
    fault_names_df = pd.read_csv("../data/fault_names.csv",index_col="id")
    df = df[columns]
    df["fault"] = df["fault"].apply(lambda faultId: fault_names_df.loc[faultId].values[0] if faultId != -1 else "unknown")

    df['x'], df['y'], df['z'] = zip(*df.apply(convert_coordinates, axis=1))
    df.time = pd.to_datetime(df.time, infer_datetime_format=True)
    df.drop(["latitude", "longitude"], axis=1, inplace=True)
    df = df.set_index("id")
    df = df.sort_values(by="time")

    return df


def convert_coordinates(row, radius=6371):
    x = radius * math.cos(row['latitude'] * math.pi / 180) * math.cos(row['longitude'] * math.pi / 180)
    y = radius * math.cos(row['latitude'] * math.pi / 180) * math.sin(row['longitude'] * math.pi / 180)
    z = radius * math.sin(row['latitude'] * math.pi / 180)

    return x, y, z


def plot_classes(labels, lon, lat, alpha=0.5, edge='k'):
    """Plot seismic events using Mollweide projection.
    Arguments are the cluster labels and the longitude and latitude
    vectors of the events"""
    img = imread("Mollweide_projection_SW.jpg")
    plt.figure(figsize=(16, 8), frameon=False)
    x = lon / 180 * np.pi
    y = lat / 180 * np.pi
    ax = plt.subplot(111, projection="mollweide")
    print(ax.get_xlim(), ax.get_ylim())
    t = ax.transData.transform(np.vstack((x, y)).T)
    print(np.min(np.vstack((x, y)).T, axis=0))
    print(np.min(t, axis=0))
    clims = np.array([(-np.pi, 0), (np.pi, 0), (0, -np.pi / 2), (0, np.pi / 2)])
    lims = ax.transData.transform(clims)
    plt.close()
    plt.figure(figsize=(16, 8), frameon=False)
    plt.subplot(111)
    plt.imshow(img, zorder=0, extent=[lims[0, 0], lims[1, 0], lims[2, 1], lims[3, 1]], aspect=1)
    x = t[:, 0]
    y = t[:, 1]
    nots = np.zeros(len(labels)).astype(bool)
    diffs = np.unique(labels)
    ix = 0
    for lab in diffs[diffs >= 0]:
        mask = labels == lab
        nots = np.logical_or(nots, mask)
        plt.plot(x[mask], y[mask], 'o', markersize=4, mew=1, zorder=1, alpha=alpha, markeredgecolor=edge)
        ix = ix + 1
    mask = np.logical_not(nots)
    if np.sum(mask) > 0:
        plt.plot(x[mask], y[mask], '.', markersize=1, mew=1, markerfacecolor='w', markeredgecolor=edge)
    plt.axis('off')
