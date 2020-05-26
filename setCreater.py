from captcha.image import ImageCaptcha
import numpy as np
import random
from tensorflow.keras.utils import Sequence
from PIL import Image


class CaptchaSequence(Sequence):
    def __init__(self, characters, batch_size, steps, n_len=4, width=128, height=64):
        self.characters = characters
        self.batch_size = batch_size
        self.steps = steps
        self.n_len = n_len
        self.width = width
        self.height = height
        self.n_class = len(characters)
        self.generator = ImageCaptcha(width=width, height=height)

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        y = [np.zeros((self.batch_size, self.n_class), dtype=np.uint8) for i in range(self.n_len)]
        for i in range(self.batch_size):
            random_str = ''.join([random.choice(self.characters) for j in range(self.n_len)])
            X[i] = np.array(self.generator.generate_image(random_str)) / 255.0
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, self.characters.find(ch)] = 1
        return X, y


def get_set(characters, width, height):
    train_data = CaptchaSequence(characters, batch_size=128, steps=1000, width=width, height=height)
    valid_data = CaptchaSequence(characters, batch_size=128, steps=100, width=width, height=height)
    return train_data, valid_data


def png2X(png_file, width, height):
    X = np.zeros((1, height, width, 3), dtype=np.float32)
    img = np.array(Image.open(png_file).resize((width, height)))
    X[0] = img / 255.0
    return X
