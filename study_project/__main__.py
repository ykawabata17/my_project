import matplotlib.pyplot as plt
from statistics import mean

from mylib.calc_shap import ShapCreate
from mylib.load_data import LoadData


def main():
    data_loader = LoadData()
    trainX, _ = data_loader.load_train_org()
    trainX_shap, _ = data_loader.load_train_shap()
    variances_org = []
    variances_shap = []
    for i in range(100):
        img_org = trainX[i]
        img_shap = trainX_shap[i]
        img_org = img_org.reshape(1, 28, 28, 1)
        img_shap = img_shap.reshape(1, 28, 28, 1)
        shap_create = ShapCreate()

        shap_create.shap_calc(img_org)
        variance_org = shap_create.variance_max_shap()
        variances_org.append(variance_org)

        shap_create.shap_calc(img_shap)
        variance_shap = shap_create.variance_max_shap()
        variances_shap.append(variance_shap)
        print(i)

    print(variances_org)
    print(variances_shap)
    print(mean(variances_org))
    print(mean(variances_shap))
    x = range(len(variances_org))
    plt.plot(x, variances_org, marker="o", color="red", linestyle="--")
    plt.plot(x, variances_shap, marker="v", color="blue", linestyle=":")
    plt.show()


if __name__ == '__main__':
    main()
