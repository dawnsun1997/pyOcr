# captcha  matplotlib numpy Tensorflow
# ypwhs/captcha_break
from tensorflow.keras.utils import Sequence
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import string
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import *

import setCreater

characters = string.digits + string.ascii_uppercase  # 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
width, height, n_len, n_class = 128, 64, 4, len(characters)


def main():
    input_tensor = Input((height, width, 3))
    x = input_tensor
    for i, n_cnn in enumerate([2, 2, 2, 2, 2]):
        for j in range(n_cnn):
            x = Conv2D(32 * 2 ** min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        x = MaxPooling2D(2)(x)

    x = Flatten()(x)
    x = [Dense(n_class, activation='softmax', name='c%d' % (i + 1))(x) for i in range(n_len)]
    model = Model(inputs=input_tensor, outputs=x)
    train_data, valid_data = setCreater.get_set(characters, width=width, height=height)

    # print('new')
    # callbacks = [EarlyStopping(patience=3), CSVLogger('cnn.csv'),
    #              ModelCheckpoint('cnn_best.h5', save_best_only=True)]
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=Adam(1e-4, amsgrad=True),
    #               metrics=['accuracy'])
    # model.fit_generator(train_data, epochs=100, validation_data=valid_data, workers=1, use_multiprocessing=False,
    #                     callbacks=callbacks)

    print('load')
    model.load_weights('cnn_best.h5')
    callbacks = [EarlyStopping(patience=3), CSVLogger('cnn.csv', append=True),
                 ModelCheckpoint('cnn_best.h5', save_best_only=True)]
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-4, amsgrad=True),
                  metrics=['accuracy'])

    # print('con')
    # model.fit_generator(train_data, epochs=100, validation_data=valid_data, workers=1, use_multiprocessing=False,
    #                     callbacks=callbacks)

    def decode(y):
        y = np.argmax(np.array(y), axis=2)[:, 0]
        return ''.join([characters[x] for x in y])

    data = setCreater.CaptchaSequence(characters, batch_size=1, steps=1, width=width, height=height)
    X, y = data[0]
    # X = setCreater.png2X(r"C:\Users\vvave\desktop\ocr\png\5.jpg", 128, 64)
    y_pred = model.predict(X)
    plt.title('real: %s\npred:%s' % (decode(y), decode(y_pred)))
    plt.imshow(X[0], cmap='gray')
    plt.axis('off')
    print('real: %s\npred: %s' % (decode(y), decode(y_pred)))


if __name__ == '__main__':
    main()
