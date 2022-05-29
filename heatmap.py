import glob
import json
import os

import optuna

from study_project.mylib.utils import get_home_path, data_set
from study_project.mylib.utils import classification_scorer, SupervisedUMAP


PATH = get_home_path()


def main():
    map_datas = glob.glob(PATH + 'data/shap_all/*_shap.json')
    for map_data in map_datas:
        file_name = os.path.splitext(os.path.basename(map_data))[0]
        with open(map_data, 'r') as f:
            decode_data = json.load(f)
        dataX, dataY = data_set(decode_data)
        print("データ読み込み完了")
        objective = SupervisedUMAP(dataX, dataY, classification_scorer, file_name)
        study = optuna.create_study(direction="minimize")
        print("学習開始")        
        study.optimize(objective, n_trials=100)


if __name__=='__main__':
    main()