import numpy as np
import librosa
import os
import os.path
import matplotlib.pyplot as plt
import gc

from keras import optimizers
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical
from tensorflow.python.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

DATA_PATH = "./data/"

def wav2mfcc(file_path, max_pad_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)

def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)


def save_data_to_array(path=DATA_PATH, max_pad_len=11):
    labels, _, _ = get_labels(path)

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in wavfiles:
            try:
                mfcc = wav2mfcc(wavfile, max_pad_len=max_pad_len)
                mfcc_vectors.append(mfcc)
            except:
                print(wavfile) #Print out negative values
        np.save(label + '.npy', mfcc_vectors)





def get_train_test(split_ratio=0.8, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)

    # Getting first arrays
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)
    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)


save_data_to_array()

X_train, X_test, y_train, y_test = get_train_test()

print(X_train.shape)
# (18811, 20, 11)

X_train = X_train.reshape(X_train.shape[0], 20, 11, 1)
# print(X_train.shape)
# (18811, 20, 11, 1)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)


X_train, X_test, y_train, y_test = get_train_test()

X_train = X_train.reshape(X_train.shape[0], 20, 11, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

sgd = optimizers.SGD(lr=0.1, clipnorm=1)

model = Sequential()
# -------------------CNN---------------------------
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
# model.add(Dense(2048, activation='relu'))
# model.add(Dropout(0.10))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.20))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Dense(114, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=sgd,
              metrics=['mae','acc'])

model.summary()

if not os.path.exists('model/cnn_model.ckpt'):
    history = model.fit(X_train, y_train_hot, batch_size=64, epochs=200
                        , verbose=2, validation_data=(X_test, y_test_hot))
    pyplot.plot(history.history['val_acc'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    acute = "%.2f" % (max(history.history['val_acc'])*100), '%'
    wer = ("%.2f" % ((1-max(history.history['val_acc']))*100)), '%'
    plt.title(("Best Accuracy: ", acute, 'Word Error Rate: ', wer))
    plt.legend()
    pyplot.show()
    model.save('model/cnn_model.ckpt')
else:
    contModel = load_model('model/cnn_model.ckpt')
    testScore = contModel.evaluate(X_train, y_train_hot, verbose=0)
    print("%s: %.2f%%" % (contModel.metrics_names[0], testScore[0] * 100))