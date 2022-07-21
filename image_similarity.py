import glob

import cv2
import numpy as np
from tensorflow.keras.datasets import mnist

from study_project.mylib.utils import  get_home_path


PATH = get_home_path()


(trainX_org, trainY_org), _ = mnist.load_data()

for i in range(10):
    files = glob.glob(PATH + f'images/shap_train_redef_bg/{i}/*.jpg')
    for file in files:
        img_noise = cv2.imread(file)
        for img_org in trainX_org:
            img_org = img_org.reshape(28,28)

