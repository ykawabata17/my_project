import glob
import json
import os
import re

import cv2
from matplotlib import pyplot as plt
import numpy as np
import umap.umap_ as umap
import umap.plot
import optuna

from study_project.mylib.load_data import LoadData
from study_project.mylib.calc_shap import ShapCreate
from study_project.mylib.utils import normalization_list, get_home_path, data_set
from study_project.mylib.utils import SupervisedUMAP, classification_scorer


PATH = get_home_path()


def main():
    data_loader = LoadData()
    trainX_org, trainY_org = data_loader.load_train_org(shuffle=True)
    trainX_org = trainX_org.reshape(60000, 784)
    print(trainY_org)
    objective = SupervisedUMAP(
        trainX_org, trainY_org, classification_scorer, 'original_image')
    study = optuna.create_study(direction="minimize")
    print("学習開始")
    study.optimize(objective, n_trials=100)


if __name__ == "__main__":
    main()
