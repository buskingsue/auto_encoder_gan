import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist

input_img = Input(shape=(28, 28, 1))
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
#28 * 28
x = MaxPool2D((2, 2), padding='same')(x)
#14 * 14
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#14 * 14
x = MaxPool2D((2, 2), padding='same')(x)
#7 * 7
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#7 * 7
encoded = MaxPool2D((2, 2), padding='same')(x)
#4 * 4
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
# 4 * 4
x = UpSampling2D((2, 2))(x)
# 8 * 8
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# 8 * 8
x = UpSampling2D((2, 2))(x)
# 16 * 16
x = Conv2D(16, (3, 3), activation='relu')(x)
# 14 * 14
x = UpSampling2D((2, 2))(x)
# 28 * 28
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
# 14 * 14

autoencoder = Model(input_img, decoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255
# print(x_train.shape)
# print(x_train[0])
conv_x_train = x_train.reshape(-1, 28, 28, 1)
conv_x_test = x_test.reshape(-1, 28, 28, 1)
# print(conv_x_train.shape)
# print(conv_x_train)

noise_factor = 0.5
conv_x_train_noisy = conv_x_train + np.random.normal(0, 1, size=conv_x_train.shape) * noise_factor
conv_x_train_noisy = np.clip(conv_x_train_noisy, 0.0, 1.0)
conv_x_test_noisy = conv_x_test + np.random.normal(0, 1, size=conv_x_test.shape) * noise_factor
conv_x_test_noisy = np.clip(conv_x_test_noisy, 0.0, 1.0)
plt.figure(figsize=(20, 4))
n = 10
for i in range(n):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 1 + n)
    plt.imshow(conv_x_test_noisy[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

fit_hist = autoencoder.fit(conv_x_train_noisy, conv_x_train, epochs=50,
            batch_size=256, validation_data=(conv_x_test_noisy, conv_x_test))
autoencoder.save('./models/autoencoder_noisy.h5')

decoded_img = autoencoder.predict(conv_x_test[:10])



plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(conv_x_test_noisy[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 1 + n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.show()











