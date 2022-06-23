import glob
import json
import os
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from umap.parametric_umap import ParametricUMAP

from study_project.mylib.utils import get_home_path, data_set

PATH = get_home_path()

map_datas = glob.glob(PATH + 'data/shap_all/org_org.json')
for map_data in map_datas:
    with open(map_data, 'r') as f:
        decode_data = json.load(f)
    dataX, dataY = data_set(decode_data)
    print("データ読み込み完了")
dataX = np.array(dataX)
print(dataX.shape)
dataX = dataX.reshape(3000, 7840)

dims = (10, 28, 28)
n_components = 2
encoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=dims),
    tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, strides=(2, 2), activation="relu", padding="same"
    ),
    tf.keras.layers.Conv2D(
        filters=64, kernel_size=3, strides=(2, 2), activation="relu", padding="same"
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation="relu"),
    tf.keras.layers.Dense(units=256, activation="relu"),
    tf.keras.layers.Dense(units=n_components),
])

embedder = ParametricUMAP(encoder=encoder, dims=dims)
embedder.save(PATH + 'data/')
embedding = embedder.fit_transform(dataX)

x = embedding[:, 0]
y = embedding[:, 1]

plt.figure()
for n in np.unique(dataY):
    plt.scatter(x[dataY == n], y[dataY == n], label=n)
plt.grid()
plt.legend()
plt.show()

dataY = np.array(dataY)
fig, ax = plt.subplots(figsize=(8, 8))
sc = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=dataY.astype(int),
    cmap="tab10",
    s=0.1,
    alpha=0.5,
    rasterized=True,
)
ax.axis('equal')
ax.set_title("Parametric UMAP embedding", fontsize=20)
plt.colorbar(sc, ax=ax)
plt.show()


