import json
import optuna
import scipy
import numpy as np
from matplotlib import pyplot as plt
from os.path import expanduser, basename, splitext

from tensorflow.keras.models import load_model
import umap.umap_ as umap

from .load_data import LoadData


def get_home_path():
    return expanduser("~") + '/study_data/'


PATH = get_home_path()


def data_set(map_data):
    dataX = []
    dataY = []
    for k, v in map_data.items():
        for data in v:
            dataX.append(data)
            dataY.append(int(k))
    return dataX, dataY

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


def normalization_list(data, max_value=1, min_value=0):
    data_max = max(data)
    data_min = min(data)
    norm_data = []
    for value in data:
        y = ((value-data_min) / (data_max-data_min)) * \
            (max_value-min_value) + min_value
        norm_data.append(y)
    return norm_data


def dim_reduction_umap(data,
                       random_state=0,
                       n_neighbor=50,
                       metric='euclidean',
                       min_dist=0.1):
    """
    UMAPを用いて次元削減し、プロットする点を配列で返す関数
    data: shape(データ数, 削減したい次元数)
    return x, y: (配列)
    """
    mapper = umap.UMAP(n_components=2,
                       n_neighbors=n_neighbor,
                       random_state=random_state,
                       metric=metric,
                       min_dist=min_dist)
    mapper.fit(data)
    embedding = mapper.transform(data)
    x = embedding[:, 0]
    y = embedding[:, 1]
    return x, y


def map_separate_plot(map_datas):
    """
    それぞれの画像に対してUMAPを用いて次元削減し、
    それぞれの画像の点をプロットして2次元のマップを作成する関数
    """
    for map_data in map_datas:
        plt.figure()
        filename = splitext(basename(map_data))[0]
        with open(map_data, 'r') as f:
            data_dict = json.load(f)
        for label, data in data_dict.items():
            dataX = np.array(data)
            dataY = [int(label) for _ in range(len(data))]
            dataY = np.array(dataY)
            x, y = dim_reduction_umap(dataX)
            for n in np.unique(dataY):
                plt.scatter(x[dataY == n], y[dataY == n], label=n)
        plt.grid()
        plt.legend()
        plt.savefig(PATH + f'plot/{filename}.png')

def map_plot(map_datas):
    """
    すべての画像のデータを次元削減し、2次元のマップを作成する関数
    """
    for map_data in map_datas:
        plt.figure()
        filename = splitext(basename(map_data))[0]
        with open(map_data, 'r') as f:
            data_dict = json.load(f)
        dataX, dataY = data_set(data_dict)
        dataX = np.array(dataX)
        dataY = np.array(dataY)
        x, y = dim_reduction_umap(dataX)
        for n in np.unique(dataY):
            plt.scatter(x[dataY == n], y[dataY == n], label=n)
        plt.grid()
        plt.legend()
        plt.savefig(PATH + f'plot/{filename}_2.png')

def model_data_load(model_name, data_name):
    """
    model: (org, prop, at, hybrid)
    data: (org, shap, ae, random, test)
    """
    # モデルの読み込み
    if model_name == 'org':
        model = load_model(PATH + 'models/org/org20000.h5')
    elif model_name == 'prop':
        model = load_model(PATH + 'models/org_shap/org10000_shap10000.h5')
    elif model_name == 'at':
        model = load_model(PATH + 'models/org_ae/org10000_ae10000.h5')
    elif model_name == 'hybrid':
        model = load_model(
            PATH + 'models/org_shap_ae/org10000_shap5000_ae5000.h5')
    # データの読み込み
    data_loader = LoadData()
    if data_name == 'org':
        dataX, dataY = data_loader.load_test_org(shuffle=False)
    elif data_name == 'shap':
        dataX, dataY = data_loader.load_test_shap(shuffle=False)
    elif data_name == 'ae':
        dataX, dataY = data_loader.load_test_ae(shuffle=False)
    elif data_name == 'random':
        dataX, dataY = data_loader.load_test_random(shuffle=False)
    elif data_name == 'test':
        dataX, dataY = data_loader.load_test_data(shuffle=False)

    return model, dataX, dataY




def classification_scorer(X, Y, alpha=1e-3):
    sum = 0
    n1 = 0
    for x1, y1 in zip(X, Y):
        n1 += 1
        n2 = 0
        for x2, y2 in zip(X, Y):
            n2 += 1
            if n1 > n2:
                dist = ((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2) + 1e-53
                if y1 != y2:
                    sum += 1 / dist
                else:
                    pass

    return sum / (len(Y) * (len(Y) - 1) / 2)


class SupervisedUMAP:
    def __init__(self, X, Y, scorer):
        self.X = X
        self.Y = Y
        self.scorer = scorer
        self.best_score = 1e53
        self.best_model = None

    def __call__(self, trial):
        n_neighbors = trial.suggest_int("n_neighbors", 2, 100)
        min_dist = trial.suggest_uniform("min_dist", 0.0, 0.99)
        metric = trial.suggest_categorical("metric",
                                           ["euclidean", "manhattan", "chebyshev", "minkowski", "canberra", 
                                            "braycurtis", "mahalanobis", "cosine", "correlation"])

        mapper = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric
        )
        mapper.fit(self.X)
        embedding = mapper.transform(self.X)
        score = self.scorer(scipy.stats.zscore(embedding), self.Y)

        if self.best_score > score:
            self.best_score = score
            self.best_model = mapper

            print(self.best_model)
            title = 'trial={0}, score={1:.3e}'.format(trial.number, score)
            plt.title(title)
            for n in np.unique(self.Y):
                plt.scatter(embedding[self.Y == n], embedding[self.Y == n], label=n)
            plt.grid()
            plt.legend()
            plt.show()

        return score