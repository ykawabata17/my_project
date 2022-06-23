import glob
import json

import numpy as np

from study_project.mylib.utils import get_home_path, data_set

PATH = get_home_path()

map_datas = glob.glob(PATH + 'data/shap_all/org_org.json')
for map_data in map_datas:
    with open(map_data, 'r') as f:
        decode_data = json.load(f)
    dataX, dataY = data_set(decode_data)
    print("データ読み込み完了")
print(len(dataX))
print(np.array(dataX).shape)