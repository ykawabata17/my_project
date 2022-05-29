import numpy as np

from mylib.calc_shap import ShapCreate
from mylib.load_data import LoadData
from mylib.utils import get_home_path


PATH = get_home_path()


def main():
    models = ['org', 'prop', 'at', 'hybrid']
    datas = ['shap', 'ae']
    for model in models:
        for data in datas:
            ShapCreate.create_heatmap(model, data)
            ShapCreate.heatmap_all_sum(model, data)
            ShapCreate.heatmap_to_umap(model, data)


if __name__ == '__main__':
    main()
