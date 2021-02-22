# https://deeplearningcourses.com/c/data-science-deep-learning-in-theano-tensorflow
# https://www.udemy.com/data-science-deep-learning-in-theano-tensorflow
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

from keras.models import Sequential
from keras.layers import Dense, Activation
from util import get_normalized_data, y2indicator, get_data_train_test

import matplotlib.pyplot as plt


# NOTE: do NOT name your file keras.py because it will conflict
# with importing keras

# installation is easy! just the usual "sudo pip(3) install keras"

def ANN(dataset_path):
    # get the data, same as Theano + Tensorflow examples
    # no need to split now, the fit() function will do it
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data(dataset_path)
    # get shapes
    N, D = Xtrain.shape

    K = len(set(Ytrain))
    # by default Keras wants one-hot encoded labels
    # there's another cost function we can use
    # where we can just pass in the integer labels directly
    # just like Tensorflow / Theano
    Ytrain = y2indicator(Ytrain)
    Ytest = y2indicator(Ytest)

    # the model will be a sequence of layers
    model = Sequential()
    model2 = Sequential()
    model3 = Sequential()
    model4 = Sequential()

    # ANN with layers [30] -> [500] -> [300] -> [2]
    model.add(Dense(units=10, input_dim=D))
    model.add(Activation('relu'))
    model.add(Dense(units=5))  # don't need to specify input_dim
    model.add(Activation('relu'))
    model.add(Dense(units=K + 1))
    model.add(Activation('softmax'))

    model2.add(Dense(units=50, input_dim=D))
    model2.add(Activation('relu'))
    model2.add(Dense(units=10))  # don't need to specify input_dim
    model2.add(Activation('relu'))
    model2.add(Dense(units=K + 1))
    model2.add(Activation('softmax'))

    model3.add(Dense(units=100, input_dim=D))
    model3.add(Activation('relu'))
    model3.add(Dense(units=50))  # don't need to specify input_dim
    model3.add(Activation('relu'))
    model3.add(Dense(units=K + 1))
    model3.add(Activation('softmax'))

    model4.add(Dense(units=500, input_dim=D))
    model4.add(Activation('relu'))
    model4.add(Dense(units=100))  # don't need to specify input_dim
    model4.add(Activation('relu'))
    model4.add(Dense(units=K + 1))
    model4.add(Activation('softmax'))

    # list of losses: https://keras.io/losses/
    # list of optimizers: https://keras.io/optimizers/
    # list of metrics: https://keras.io/metrics/
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    model2.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    model3.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    model4.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # note: multiple ways to choose a backend
    # either theano, tensorflow, or cntk
    # https://keras.io/backend/

    # gives us back a <keras.callbacks.History object at 0x112e61a90>
    r = model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=100, batch_size=100)
    r2 = model2.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=100, batch_size=100)
    r3 = model3.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=100, batch_size=100)
    r4 = model4.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=100, batch_size=100)
    print("Returned:", r)

    # print the available keys
    # should see: dict_keys(['val_loss', 'acc', 'loss', 'val_acc'])
    print(r.history.keys())

    # plot some data
    plt.title("loss vs val_loss")
    plt.xlabel('epochs')
    leg = plt.legend()
    # get the lines and texts inside legend box
    leg_lines = leg.get_lines()
    leg_texts = leg.get_texts()
    plt.setp(leg_lines, linewidth=4)
    plt.setp(leg_texts, fontsize='x-large')
    plt.plot(r.history['loss'], label='loss-L10-5')
    plt.plot(r.history['val_loss'], label='val_loss-L10-5')
    plt.plot(r2.history['loss'], label='loss-L50-10')
    plt.plot(r2.history['val_loss'], label='val_loss-L50-10')
    plt.plot(r3.history['loss'], label='loss-L100-50')
    plt.plot(r3.history['val_loss'], label='val_loss-L100-50')
    plt.plot(r4.history['loss'], label='loss-L500-100')
    plt.plot(r4.history['val_loss'], label='val_loss-L500-100')
    plt.legend()
    plt.show()

    # accuracies
    plt.title("Accuracy: train vs validation")
    plt.xlabel('epochs')
    leg = plt.legend()
    # get the lines and texts inside legend box
    leg_lines = leg.get_lines()
    leg_texts = leg.get_texts()
    plt.setp(leg_lines, linewidth=4)
    plt.setp(leg_texts, fontsize='x-large')
    plt.plot(r.history['accuracy'], label='acc-L10-5')
    plt.plot(r.history['val_accuracy'], label='val_acc-L10-5')
    plt.plot(r2.history['accuracy'], label='acc-L50-10')
    plt.plot(r2.history['val_accuracy'], label='val_acc-L50-10')
    plt.plot(r3.history['accuracy'], label='acc-L100-50')
    plt.plot(r3.history['val_accuracy'], label='val_acc-L100-50')
    plt.plot(r4.history['accuracy'], label='acc-L500-100')
    plt.plot(r4.history['val_accuracy'], label='val_acc-L500-100')
    plt.legend()
    plt.show()


