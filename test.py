import cv2
import numpy as np

from study_project.mylib.calc_shap import ShapCreate
from study_project.mylib.utils import get_home_path, model_data_load

PATH = get_home_path()

model, dataX, dataY = model_data_load('org', 'org')
print("データ読み込み完了")
shap_creater = ShapCreate(model)
img = dataX[0].reshape(1, 28, 28, 1)
print(np.array(img).shape)
noise_image = shap_creater.add_noise(img)
noise_image = np.array(noise_image).reshape(1, 28, 28, 1)*255
print(np.array(noise_image).shape)
print(noise_image)
cv2.imwrite("aa.png", noise_image)
cv2.imshow('image', noise_image)
cv2.waitKey(0)