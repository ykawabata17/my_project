import numpy as np
import json

from mylib.calc_shap import ShapCreate
from mylib.load_data import LoadData
from mylib.utils import data_set_to_dict


def main():
    data_loader = LoadData()
    # umap_model = umap.UMAP(n_components=2, random_state=0, n_neighbors=5)
    testX_shap, testY_shap = data_loader.load_test_shap(shuffle=False)
    print(type(testY_shap))
    print(testY_shap.shape)
    data_dict = data_set_to_dict(testX_shap, testY_shap)
    shap_create = ShapCreate(data_dict[0])
    map_dict, map_list = shap_create.plot_shap_10_dimension()
    x = []
    for k, v in map_dict.items():
        x.append(v)
    map_dice = np.array(x)

    testY_shap = testY_shap.tolist()

    # encode_map = json.dumps(map_list)
    with open('dataX.txt', 'w') as f:
        json.dump(map_list, f)
    with open('dataY.txt', 'w') as f:
        json.dump(testY_shap, f)


if __name__ == '__main__':
    main()
