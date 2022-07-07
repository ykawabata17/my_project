from mylib.calc_shap import ShapCreate
from mylib.utils import get_home_path


PATH = get_home_path()


def main():
    models = ['org']
    datas = ['shap']
    for model in models:
        for data in datas:
            ShapCreate.create_heatmap(model, data)


if __name__ == '__main__':
    main()
