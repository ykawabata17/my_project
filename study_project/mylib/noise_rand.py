import numpy as np
import cv2

from tensorflow.keras.datasets import mnist


def random_noise(eps=0.3):
    _, (testX_org, testY_org) = mnist.load_data()
    i = 0
    for img in testX_org:
        img = img.reshape(28, 28)
        h, w = img.shape[:2]
        noise = np.random.randint(0, eps * 300, (h, w))
        img = img + noise
        cv2.imwrite('C:/Users/kawabata/study_data/random/{}.jpg'.format(i), img)
        i += 1
