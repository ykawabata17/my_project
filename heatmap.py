import glob
import json
from matplotlib import pyplot as plt

import optuna
import scipy
from umap import UMAP
import numpy as np

from study_project.mylib.utils import get_home_path


PATH = get_home_path()


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
        n_neighbors = trial.suggest_int("n_neighbors", 2, len(self.Y))
        min_dist = trial.suggest_uniform("min_dist", 0.0, 0.99)
        metric = trial.suggest_categorical("metric",
                                           ["euclidean"])

        mapper = UMAP(
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
            plt.scatter(embedding[:, 0], embedding[:, 1],
                        c=self.Y, alpha=0.5)
            plt.colorbar()
            plt.show()

        return score


def data_set(map_data):
    dataX = []
    dataY = []
    for k, v in map_data.items():
        for data in v:
            dataX.append(data)
            dataY.append(int(k))
    return dataX, dataY


def main():
    print("start")
    map_datas = glob.glob(PATH + 'data/shap_all/*_shap.json')
    for map_data in map_datas:
        print(map_data)
        with open(map_data, 'r') as f:
            decode_data = json.load(f)
        dataX, dataY = data_set(decode_data)
        # print("データ読み込み完了")

        # objective = SupervisedUMAP(dataX, dataY, classification_scorer)
        # study = optuna.create_study(direction="minimize")
        # study.optimize(objective, n_trials=100)

        # 次元削減する
        mapper = UMAP(random_state=0)
        embedding = mapper.fit_transform(dataX)

        # 結果を二次元でプロットする
        embedding_x = embedding[:, 0]
        embedding_y = embedding[:, 1]
        for n in np.unique(dataY):
            plt.scatter(embedding_x[dataY == n],
                        embedding_y[dataY == n],
                        label=n)

        # グラフを表示する
        plt.grid()
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
