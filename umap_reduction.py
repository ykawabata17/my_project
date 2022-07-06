import glob
import json
import os
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from umap.parametric_umap import ParametricUMAP
from umap.parametric_umap import load_ParametricUMAP
from tensorflow.keras.models import load_model

from study_project.mylib.load_data import LoadData
from study_project.mylib.calc_shap import ShapCreate
from study_project.mylib.utils import get_home_path, data_set, normalization_list

PATH = get_home_path()

def add_data(add_data):
    model = load_model(PATH + 'models/org_org/org20000.h5')
    add_dataY = []
    add_dataX = []
    shap_creater = ShapCreate(model)
    data_loader = LoadData()
    if add_data == 'shap':
        addX, addY = data_loader.load_test_shap(shuffle=False)
    elif add_data == 'ae':
        addX, addY = data_loader.load_test_ae(shuffle=True)
    elif add_data == 'org':
        addX, addY = data_loader.load_test_org(shuffle=True)
    elif add_data == 'random':
        addX, addY = data_loader.load_test_random(shuffle=True)
    else:
        # 写真単体の時の処理を書く
        pass
    for i in range(len(addX)):
        label = int([np.where(addY[i] == 1.0)][0][0][0])
        if label == 0:   
            add_dataY.append(str(label)+'_shap')
            img = addX[i].reshape(1, 28, 28, 1)
            # shap_value = shap_creater.shap_calc(img)['shap_values'].tolist()
            # add_dataX.append(shap_value)
            shap_sum = shap_creater.shap_calc(img)['shap_sum']
            shap_sum_norm = normalization_list(shap_sum, 1, 0)
            add_dataX.append(shap_sum_norm)
    add_dataX, add_dataY = np.array(add_dataX), np.array(add_dataY)
    random = np.arange(len(add_dataX))
    np.random.shuffle(random)
    add_dataX, add_dataY = add_dataX[random], add_dataY[random]
    return add_dataX, add_dataY

# 従来モデル/元画像のshap値を取得
map_datas = glob.glob(PATH + 'data/shap_all/org_org_norm.json')
for map_data in map_datas:
    with open(map_data, 'r') as f:
        decode_data = json.load(f)
    dataX, dataY = data_set(decode_data)
    print("データ読み込み完了")
dataX = np.array(dataX)
print(dataX.shape)
dataX = dataX.reshape(len(dataX), 10)

# 従来モデル/shap画像のshap値を取得
add_dataX, add_dataY = add_data('shap')
add_dataX = add_dataX.reshape(len(add_dataX), 10)

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

embedder = ParametricUMAP(verbose=True, autoencoder_loss=True)
embedder.save(PATH + 'data/')
# embedder = load_ParametricUMAP(PATH + 'data')
embedding = embedder.fit_transform(dataX)
additional_embedding = embedder.fit_transform(add_dataX)

x = embedding[:, 0]
y = embedding[:, 1]
add_x = additional_embedding[:, 0]
add_y = additional_embedding[:, 1]

plt.figure()
for n in np.unique(dataY):
    plt.scatter(x[dataY == n], y[dataY == n], label=n, zorder=1)
for n in np.unique(add_dataY):
    plt.scatter(add_x[add_dataY == n], add_y[add_dataY == n], label=n, 
                marker='*', zorder=2)
plt.grid()
plt.legend()
plt.show()

# dataY = np.array(dataY)
# fig, ax = plt.subplots(figsize=(8, 8))
# sc = ax.scatter(
#     embedding[:, 0],
#     embedding[:, 1],
#     c=dataY.astype(int),
#     cmap="tab10",
#     s=0.1,
#     alpha=0.5,
#     rasterized=True,
# )
# ax.axis('equal')
# ax.set_title("Parametric UMAP embedding", fontsize=20)
# plt.colorbar(sc, ax=ax)
# plt.show()


