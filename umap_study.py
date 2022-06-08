import glob
import json
import os

from matplotlib import pyplot as plt
import numpy as np
import optuna
import scipy
import umap.umap_ as umap

from study_project.mylib.utils import get_home_path, data_set


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
    def __init__(self, X, Y, scorer, file_name):
        self.X = X
        self.Y = Y
        self.scorer = scorer
        self.best_score = 1e53
        self.best_model = None
        self.folder_name = file_name

    def __call__(self, trial):
        n_neighbors = trial.suggest_int("n_neighbors", 2, 100)
        min_dist = trial.suggest_uniform("min_dist", 0.0, 0.99)
        metric = trial.suggest_categorical("metric", 
                                           ["euclidean", "manhattan", "chebyshev", "minkowski", "canberra", 
               "braycurtis", "mahalanobis", "cosine", "correlation"])
        # para_history = {}
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
            title += f'\nn_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}'

            plt.figure()
            plt.title(title)
            for n in np.unique(self.Y):
                plt.scatter(embedding[:, 0][self.Y == n],
                            embedding[:, 1][self.Y == n], label=n)
            plt.grid()
            plt.legend()
            plt.savefig(
                PATH + f'plot/study_history_sum_norm/{self.folder_name}/{trial.number}.png')
        return score


def main():
    map_datas = glob.glob(PATH + 'data/shap_sum_norm/*.json')
    for map_data in map_datas:
        file_name = os.path.splitext(os.path.basename(map_data))[0]
        with open(map_data, 'r') as f:
            decode_data = json.load(f)
        dataX, dataY = data_set(decode_data)
        print(dataX.shape)
        print("データ読み込み完了")
        objective = SupervisedUMAP(
            dataX, dataY, classification_scorer, file_name)
        study = optuna.create_study(direction="minimize")
        print("学習開始")
        study.optimize(objective, n_trials=100)


if __name__ == '__main__':
    main()
