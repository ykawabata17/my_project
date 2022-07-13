from mylib.calc_shap import ShapCreate
from mylib.utils import get_home_path, model_data_load


PATH = get_home_path()


def main():
    model, dataX, dataY = model_data_load('org', 'train_org')
    shap_creater = ShapCreate(model)
    shap_creater.save_noise_image(dataX, dataY)


if __name__ == '__main__':
    main()
