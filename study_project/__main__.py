from mylib.calc_shap import ShapCreate
from tensorflow.keras.datasets import mnist


def main():
    (trainX, _), _ = mnist.load_data()
    shap_create = ShapCreate()
    img = trainX[0]
    img = img.astype('float32') / 255
    img = img.reshape(1, 28, 28, 1)
    shap_create.shap_calc(img)
    shap_create.create_fig()



if __name__ == '__main__':
    main()
