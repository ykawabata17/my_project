from multiprocessing import reduction
import numpy as np
from sklearn.mainfold import TSNE
import matplotlib.pyplot as plt


def data_set_to_dict(dataX, dataY):
    data_dict = {}
    for i in range(len(dataX)):
        ans = int([np.where(dataY[i] == 1.0)][0][0][0])
        if ans in data_dict:
            data_dict[ans].append(dataX[i])
        else:
            data_dict[ans] = [dataX[i]]
    for k, v in data_dict.items():
        data_dict[k] = np.array(v)
    return data_dict


def dimensional_reduction_umap(data):
    pass


def dimensional_reduction_tsne(data):
    model_tsne = TSNE(n_components=2, preplexity=2)
    reduction_list = model_tsne.fit_transform(data)
    plt.figure(figsize=(13, 7))
    plt.scatter(reduction_list[:, 0], reduction_list[:, 1],
                c=y, cmap='jet',
                s=15, alpha=0.5)
    plt.axis('off')
    plt.colorbar()
    plt.show()
