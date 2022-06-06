from ast import Load
import glob
import json
import os
import re

import cv2
from matplotlib import pyplot as plt
import numpy as np
import umap.umap_ as umap
from tensorflow.keras.models import load_model

from study_project.mylib.load_data import LoadData
from study_project.mylib.calc_shap import ShapCreate
from study_project.mylib.utils import normalization_list, get_home_path, data_set


PATH = get_home_path()


def add_plot_to_map(dataX, dataY, add_file, model, parameter):
    if add_file == 'shap':
        data_loader = LoadData()
        x, y = data_loader.load_test_shap(shuffle=True)
        x, y = x[:100], y[:100]
        shap_create = ShapCreate(model)
        for i in range(len(x)):
            img = x[i].reshape(1, 28, 28, 1)
            shap_info = shap_create.shap_calc(img)
            shap_sum = shap_info['shap_sum']
            shap_sum_norm = normalization_list(shap_sum, 1, 0)
            dataX = dataX.tolist()
            dataX.append(shap_sum_norm)
            label = int([np.where(y[i] == 1.0)][0][0][0])
            dataY = dataY.tolist()
            dataY.append(f'{label}_add')
            dataX = np.array(dataX)
            dataY = np.array(dataY)
    else:
        # add_fileの画像からshapだす
        img = cv2.imread(add_file, cv2.IMREAD_GRAYSCALE)
        label = re.sub(r"\D", "", add_file)[0]
        img = np.array(img).astype('float32') / 255
        img = img.reshape(1, 28, 28, 1)
        label = np.array(label)
        shap_create = ShapCreate(model)
        shap_info = shap_create.shap_calc(img)
        shap_sum = shap_info['shap_sum']
        add_data_shap = normalization_list(shap_sum, 1, 0)
        dataX = dataX.tolist()
        dataX.append(add_data_shap)
        dataY = dataY.tolist()
        dataY.append('add_data')
        dataX = np.array(dataX)
        dataY = np.array(dataY)

    # 追加データも加えてマップ作成
    mapper = umap.UMAP(n_components=2,
                       n_neighbors=parameter['n_neighbors'],
                       metric=parameter['metric'],
                       min_dist=parameter['min_dist'])
    embedding = mapper.fit_transform(dataX)
    x = embedding[:, 0]
    y = embedding[:, 1]
    plt.figure()
    for n in np.unique(dataY):
        if 'add' in n:
            plt.scatter(x[dataY == n], y[dataY == n],
                        label=n, color='k', marker='D')
        else:
            plt.scatter(x[dataY == n], y[dataY == n], label=n)
    plt.grid()
    plt.legend()
    plt.show()


def main(**kwargs):
    map_datas = glob.glob(PATH + 'data/shap_all/org_org.json')
    for map_data in map_datas:
        file_name = os.path.splitext(os.path.basename(map_data))[0]
        print(file_name)
        if file_name == 'at_shap':
            parameter = {'n_neighbors': 6,
                         'min_dist': 0.768167, 'metric': 'canberra'}
            model = load_model(PATH + 'models/org_ae/org10000_ae10000.h5')
        elif file_name == 'org_shap':
            parameter = {'n_neighbors': 4,
                         'min_dist': 0.927820, 'metric': 'canberra'}
            model = load_model(PATH + 'models/org/org20000.h5')
        elif file_name == 'hybrid_shap':
            parameter = {'n_neighbors': 10,
                         'min_dist': 0.782874, 'metric': 'canberra'}
            model = load_model(
                PATH + 'models/org_shap_ae/org10000_shap5000_ae5000.h5')
        elif file_name == 'prop_shap':
            parameter = {'n_neighbors': 10,
                         'min_dist': 0.934518, 'metric': 'canberra'}
            model = load_model(PATH + 'models/org_shap/org10000_shap10000.h5')
        elif file_name == 'org_org':
            parameter = {'n_neighbors': 17,
                         'min_dist': 0.500905, 'metric': 'canberra'}
            model = load_model(PATH + 'models/org_org/org20000.h5')
        with open(map_data, 'r') as f:
            decode_data = json.load(f)
        dataX, dataY = data_set(decode_data)
        img_name = kwargs['file_name']
        add_plot_to_map(dataX, dataY, img_name, model, parameter)


if __name__ == "__main__":
    import sys
    args = sys.argv
    add_file = args[1]
    main(file_name=add_file)
