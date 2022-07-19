import glob
import os
import re

import pandas as pd

from study_project.mylib.utils import  get_home_path


PATH = get_home_path()

df = pd.DataFrame(0, index=[x for x in range(10)], columns=[x for x in range(10)])

for i in range(10):
    files = glob.glob(PATH + f'images/shap_train_redef_bg/{i}/*.jpg')
    for file in files:
        file_name = os.path.basename(file)
        num = re.sub(r"\D", "", file_name)
        before = int(num[0])
        after = int(num[1])
        df[after][before] += 1

df.to_csv(PATH + 'after_redef_bg_label.csv')