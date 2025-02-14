#
import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255
# print(x_train.shape)
# print(x_train[0])
conv_x_train = x_train.reshape(-1, 28, 28, 1)
conv_x_test = x_test.reshape(-1, 28, 28, 1)
print(conv_x_train.shape)
print(conv_x_test.shape)

noise_factor = 0.3
conv_x_train_noisy = conv_x_train + np.random.normal(
loc=0.0, scale=1.0, size=conv_x_train.shape) * noise_factor
conv_x_train_noisy = np.clip(conv_x_train_noisy,  0.0,  1.0)

conv_x_test_noisy = conv_x_test + np.random.normal(
loc=0.0, scale=1.0, size=conv_x_test.shape) * noise_factor
conv_x_test_noisy = np.clip(conv_x_test_noisy,  0.0,  1.0)

autoencoder = load_model('./models/autoencoder_noisy.h5')
decoded_img = autoencoder.predict(conv_x_test_noisy[:10])



# 잡음 섞기

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

# fit_hist = autoencoder.fit(conv_x_train_noisy, conv_x_train, epochs=50,
#             batch_size=256, validation_data=(conv_x_test_noisy, conv_x_test))
#
# decoded_img = autoencoder.predict(conv_x_test[:10])
# #잡음이 섞인 이미지가 원본과 얼마나 차이가 있는지 살펴봄
#
#
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     ax = plt.subplot(2, 10, i + 1)
#     plt.imshow(conv_x_test_noisy[i].reshape(28, 28))
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     ax = plt.subplot(2, 10, i + 1 + n)
#     plt.imshow(decoded_img[i].reshape(28, 28))
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()










