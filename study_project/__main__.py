import numpy as np
import json

from mylib.calc_shap import ShapCreate
from mylib.load_data import LoadData
from mylib.utils import data_set_to_dict
from mylib.utils import normalization_list


def main():
    data_loader = LoadData()
    testX, testY = data_loader.load_test_ae(shuffle=False)
    data_dict = data_set_to_dict(testX, testY)
    testY = testY.tolist()
    for num, data in data_dict.items():
        shap_create = ShapCreate(data)
        map_dict, map_list = shap_create.plot_shap_10_dimension()
        map_norm_list = []
        for map_value in map_list:
            norm_value = normalization_list(map_value, 1, -1)
            map_norm_list.append(norm_value)
        with open(f'C:/Users/kawabata/study_data/data/従来モデル_ae_data/dataX_{num}.txt', 'w') as f:
            json.dump(map_norm_list, f)
        with open(f'C:/Users/kawabata/study_data/data/従来モデル_ae_data/dataY_{num}.txt', 'w') as f:
            json.dump(testY, f)
        print(num)


if __name__ == '__main__':
    main()
