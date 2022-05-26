import optuna
import scipy
from umap import UMAP
from sklearn import datasets
import matplotlib.pyplot as plt

dataset = datasets.load_diabetes()


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
                                           ["euclidean", "manhattan", "chebyshev", "minkowski", "canberra",
                                            "braycurtis", "mahalanobis", "cosine", "correlation"])

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

        return score


objective = SupervisedUMAP(dataset.data, dataset.target, classification_scorer)
print(dataset.data)
print(dataset.target)
print(type(dataset.target[0]))
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
