import optuna
import umap.umap_ as umap
from matplotlib import pyplot as plt
import numpy as np

from study_project.mylib.load_data import LoadData
from study_project.mylib.utils import SupervisedUMAP, classification_scorer, get_home_path


PATH = get_home_path()


def main():
    data_loader = LoadData()
    trainX_org, trainY_org = data_loader.load_train_org(shuffle=True)
    trainX_org = trainX_org.reshape(10000, 784)
    print(trainY_org)
    mapper = umap.UMAP(n_neighbors=15, min_dist=0.5, metric='canberra')
    embedding = mapper.fit_transform(trainX_org)
    x = embedding[:, 0]
    y = embedding[:, 1]
    print(x)
    print(y)
    for n in np.unique(trainY_org):
        plt.scatter(x[trainY_org == n],
                    y[trainY_org == n],
                    label=n)

    # グラフを表示する
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
