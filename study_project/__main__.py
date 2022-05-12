import numpy as np

from mylib.calc_shap import ShapCreate
from mylib.load_data import LoadData
from mylib.utils import data_set_to_dict, dimensional_reduction_tsne


def main():
    data_loader = LoadData()
    # umap_model = umap.UMAP(n_components=2, random_state=0, n_neighbors=5)
    testX_shap, testY_shap = data_loader.load_test_shap(shuffle=False)
    data_dict = data_set_to_dict(testX_shap, testY_shap)
    shap_create = ShapCreate(data_dict[0])
    map_dict = shap_create.plot_shap_10_dimension()
    x = []
    for k, v in map_dict.items():
        x.append(v)
    data = np.array(x)
    dimensional_reduction_tsne(data)


if __name__ == '__main__':
    main()
