import json
from re import I
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
import umap.umap_ as umap

from study_project.mylib.utils import normalization_list


def create_2d_umap(dataX, dataY, random_state=0, n_neighbor=50):
    """Dimension reduction to 2D by UMAP.

    Args:
        dataX (np.array): target data
        dataY (np.array): target label data
        random_state (int, optional): random seed num. Defaults to 0.
        n_neighbor (int, optional): n_neighbor num. Defaults to 15.
    """
    edited_dataY = []
    for data in dataY:
        edited_dataY.append(np.where(data == 1.0)[0][0])
    dataY = np.array(edited_dataY)
    index_count = np.bincount(dataY)
    mapper = umap.UMAP(
        n_components=2,
        random_state=random_state,
        n_neighbors=n_neighbor)
    embedding = mapper.fit_transform(dataX)
    return embedding, index_count


def create_3d_umap(dataX, dataY, random_state=0, n_neighbor=50):
    """Dimension reduction to 3D by UMAP.

    Args:
        dataX (np.array): target data
        dataY (np.array): target label data
        random_state (int, optional): random seed num. Defaults to 0.
        n_neighbor (int, optional): n_neighbor num. Defaults to 15.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    mapper = umap.UMAP(
        n_components=3,
        random_state=random_state,
        n_neighbors=n_neighbor)
    embedding = mapper.fit_transform(dataX)
    embedding_x = embedding[:, 0]
    embedding_y = embedding[:, 1]
    for n in np.unique(dataY):
        ax.scatter(embedding_x[dataY == n],
                   embedding_y[dataY == n],
                   label=n)
    plt.grid()
    plt.legend()
    plt.show()


def main():
    # データセットを読み込む
    dataset = datasets.load_digits()
    X, y = dataset.data, dataset.target
    print(y)
    # 次元削減する
    mapper = umap.UMAP(random_state=0)
    embedding = mapper.fit_transform(X)

    # 結果を二次元でプロットする
    embedding_x = embedding[:, 0]
    embedding_y = embedding[:, 1]
    for n in np.unique(y):
        plt.scatter(embedding_x[y == n],
                    embedding_y[y == n],
                    label=n)

    # グラフを表示する
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    for i in range(10):
        with open(f'C:/Users/kawabata/study_data/plot_org/dataX_{i}.txt', 'r') as f:
            decodeX_data = json.load(f)
        with open(f'C:/Users/kawabata/study_data/plot_org/dataY_{i}.txt', 'r') as f:
            decodeY_data = json.load(f)
        map_list = np.array(decodeX_data)
        dataY = np.array(decodeY_data)
        embedding, index_count = create_2d_umap(map_list, dataY, )
        dataY = [i for _ in range(index_count[i])]
        embedding_x = embedding[:, 0]
        embedding_y = embedding[:, 1]
        for n in np.unique(dataY):
            plt.scatter(embedding_x[dataY == n],
                        embedding_y[dataY == n],
                        label=n)
    plt.grid()
    plt.legend()
    plt.show()
