import json
import numpy as np

from keras.models import load_model

from mylib.calc_shap import ShapCreate
from mylib.load_data import LoadData
from mylib.utils import data_set_to_dict
from mylib.utils import normalization_list


PATH = 'C:/Users/kawabata/study_data/'


def add_plot_data(kind):
    data_loader = LoadData()
    model = load_model(PATH + 'models/org/org20000.h5')
    dataX, dataY = data_loader.load_add_data(kind=kind)
    data_dict = data_set_to_dict(dataX, dataY)
    dataX_list = []
    for label, data in data_dict.items():
        shap_create = ShapCreate(data, model)
        map_list = shap_create.plot_shap_10_dimension()
        map_norm_list = []
        for map_value in map_list:
            norm_value = normalization_list(map_value, 1, 0)
            map_norm_list.append(norm_value)
        dataX_list.append(map_norm_list)
        print(label)
    map_data = {}
    for index, data in enumerate(dataX_list):
        map_data[index] = data
    with open(PATH + f'data/add_data/org_{kind}.json', 'w') as f:
        f.write(json.dumps(map_data))


def main():
    models = ['org', 'prop', 'at', 'hybrid']
    datas = ['shap', 'ae']
    for model in models:
        for data in datas:
            ShapCreate.create_heatmap(model, data)


if __name__ == '__main__':
    main()
