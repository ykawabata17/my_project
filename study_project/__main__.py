import matplotlib.pyplot as plt
from statistics import mean
from tqdm import tqdm

from mylib.calc_shap import ShapCreate
from mylib.load_data import LoadData


def main():
    indexes_all = {}
    data_loader = LoadData()
    trainX, _ = data_loader.load_train_org()
    trainX_shap, _ = data_loader.load_train_shap()
    shap_create = ShapCreate()
    for i in tqdm(range(0, 500)):
        img_org = trainX[i]
        img_shap = trainX_shap[i]
        img_org = img_org.reshape(1, 28, 28, 1)
        img_shap = img_shap.reshape(1, 28, 28, 1)
        shap_create.shap_calc(img_shap)
        indexes, max_index = shap_create.plot_max_shap()
        if max_index not in indexes_all:
            indexes_all[max_index] = indexes
        else:
            indexes_all[max_index].extend(indexes)

    for i in range(10):
        plot_num = indexes_all[i]
        x = []
        y = []
        for index in plot_num:
            x.append(index[0])
            y.append(index[1])
        plt.xlim(0, 28)
        plt.ylim(0, 28)
        plt.scatter(x, y, marker="o")
        plt.show()


if __name__ == '__main__':
    main()
