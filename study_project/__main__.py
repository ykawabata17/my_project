from sklearn.metrics import median_absolute_error
from mylib.calc_shap import ShapCreate
from mylib.evaluate_acc import EvaluateAcc
from mylib.model import ModelCreate
from mylib.utils import get_home_path, model_data_load
import numpy as np


PATH = get_home_path()


def main():
    # model, dataX, dataY = model_data_load('org', 'test_org')
    # shap_creater = ShapCreate(model)
    # shap_creater.save_noise_image(dataX, dataY)

    acc = {'org':[], 'shap':[], 'ae':[], 'random':[]}
    for i in range(11):
        model_creater = ModelCreate()
        print(f"{i}回目")
        for j in range(11):
            print(f"{j}番目のモデル")
            num = j*1000
            model_creater.model_org_shap(10000, num)
        accs = EvaluateAcc.model_evaluate('org_shap_same_bg')
        acc['org'].append(accs[0])
        acc['shap'].append(accs[1])
        acc['ae'].append(accs[2])
        acc['random'].append(accs[3])
        
    for _, v in acc.items():
        v = np.array(v).T.tolist()
        
    for k, v_list in acc.items():
        for v in v_list:
            count = 1
            avg = sum(v) / 10
            print(f"{k}:{count}回目:{avg}")
            count += 1
        

if __name__ == '__main__':
    main()
