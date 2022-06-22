import glob
import json
import os
import re
from turtle import color

import cv2
from matplotlib import pyplot as plt
import numpy as np
import umap.umap_ as umap
import optuna
from tensorflow.keras.models import load_model

from study_project.mylib.load_data import LoadData
from study_project.mylib.calc_shap import ShapCreate
from study_project.mylib.utils import normalization_list, get_home_path, data_set
from study_project.mylib.utils import SupervisedUMAP, classification_scorer


PATH = get_home_path()


def add_data(dataX, dataY, add_data):
    model = load_model(PATH + 'models/org_org/org20000.h5')
    shap_creater = ShapCreate(model)
    data_loader = LoadData()
    if add_data == 'shap':
        addX, addY = data_loader.load_test_shap(shuffle=False)
    elif add_data == 'ae':
        addX, addY = data_loader.load_test_ae(shuffle=True)
    elif add_data == 'org':
        addX, addY = data_loader.load_test_org(shuffle=True)
    elif add_data == 'random':
        addX, addY = data_loader.load_test_random(shuffle=True)
    else:
        # 写真単体の時の処理を書く
        pass
    
    for add_label in range(0, 10):
        for i in range(len(addX)):
            label = int([np.where(addY[i] == 1.0)][0][0][0])
            if add_label == label:
                dataY.append(f'{label}_shap')
                img = addX[i].reshape(1, 28, 28, 1)
                shap_sum = shap_creater.shap_calc(img)['shap_sum']
                shap_sum_norm = normalization_list(shap_sum, 1, 0)
                dataX.append(shap_sum_norm)
        dataX, dataY = np.array(dataX), np.array(dataY)
        random = np.arange(len(dataX))
        np.random.shuffle(random)
        dataX, dataY = dataX[random], dataY[random]
        
        print('削減開始')
        objective = SupervisedUMAP(
            dataX, dataY, classification_scorer, f'add_data_{add_label}')
        study = optuna.create_study(direction="minimize")
        print("学習開始")
        study.optimize(objective, n_trials=100)


def main():
    map_data = glob.glob(PATH + 'data/shap_all_norm/org_org.json')
    print(map_data)
    with open(map_data[0], 'r') as f:
        decode_data = json.load(f)
    dataX, dataY = data_set(decode_data)
    add_data(dataX, dataY, 'shap')


if __name__ == '__main__':
    main()
