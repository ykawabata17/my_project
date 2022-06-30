import glob
import json

import csv
import pandas as pd

from study_project.mylib.utils import get_home_path, data_set

PATH = get_home_path()

map_datas = glob.glob(PATH + 'data/shap_all_norm/org_org2.json')
for map_data in map_datas:
    with open(map_data, 'r') as f:
        decode_data = json.load(f)
    dataX, dataY = data_set(decode_data)
    print("データ読み込み完了")
with open(PATH + 'csvfile.csv','w', newline="") as file:    
    writer = csv.writer(file)
    writer.writerow(["0","1","2","3","4","5","6","7","8","9", 'label'])
    writer.writerows(dataX)

df = pd.read_csv(PATH + 'csvfile.csv')
df['label'] = dataY
df.to_csv(PATH + 'output.csv')