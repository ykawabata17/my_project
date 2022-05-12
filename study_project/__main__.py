from mylib.calc_shap import ShapCreate
from mylib.load_data import LoadData
from mylib.utils import data_set_to_dict


def main():
    data_loader = LoadData()
    testX_shap, testY_shap = data_loader.load_test_shap(shuffle=False)
    data_dict = data_set_to_dict(testX_shap, testY_shap)
    shap_create = ShapCreate(data_dict[0])
    map_dict = shap_create.plot_shap_10_dimension()
    for k, v in map_dict.items():
        print(v)


if __name__ == '__main__':
    main()
