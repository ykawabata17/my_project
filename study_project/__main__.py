from keras.models import load_model

from mylib.load_data import LoadData
from mylib.grad_cam import grad_cam


def main():
    model = load_model('C:/Users/kawabata/study_data/models/original/model.h5')
    trainX_org, trainY_org = LoadData.load_train_org()
    img = trainX_org[19]
    grad_cam(model, img)


if __name__ == '__main__':
    main()
