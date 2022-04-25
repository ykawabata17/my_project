import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


def grad_cam(model, img):
    img_predict = []
    conv_layer_output = model.get_layer("conv").output
    input_val = (28, 28, 1)
    img_predict.append(np.asarray(img))
    img_predict = np.asarray(img_predict)
    prediction = model.predict(img_predict)
    prediction_idx = np.argmax(prediction)
    loss = model.get_layer("output").output[0][prediction_idx]

    g = tf.Graph()
    with g.as_default():
        grads = K.gradients(loss, conv_layer_output)[0]
    grads_func = K.function([model.input, K.learning_phase()], [conv_layer_output, grads])

    (conv_output, conv_values) = grads_func([np.asarray([input_val]), 0])
    conv_output = conv_output[0]
    conv_values = conv_values[0]

    weights = np.mean(conv_values, axis=(0, 1))
    cam = np.dot(conv_output, weights)

    # Conv層の画像はサイズが違うのでリサイズ。
    cam = cv2.resize(cam, (28, 28), cv2.INTER_LINEAR)

    # heatmap?
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    # モノクロ画像に疑似的に色をつける
    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # オリジナルイメージもカラー化
    org_img = cv2.cvtColor(np.uint8(img), cv2.COLOR_GRAY2BGR)  # (w,h) -> (w,h,3)

    # 元のイメージに合成
    rate = 0.4
    cam = cv2.addWeighted(src1=org_img, alpha=(1 - rate), src2=cam, beta=rate, gamma=0)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)  # BGR -> RGBに変換

    # 表示
    plt.imshow(cam)
    plt.show()
