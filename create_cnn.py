from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation


class CNNModel:
    @staticmethod
    def build():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same", input_shape=(28, 28, 1)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(Dense(10))
        model.add(Activation("softmax"))

        return model
