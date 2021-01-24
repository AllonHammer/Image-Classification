import tensorflow.keras as keras
import numpy as np
import os
import gzip
from matplotlib import pyplot as plt
from tensorflow.keras.regularizers import l2
#from tensorflow.keras.callbacks import EarlyStopping
from configs import  batch_size, epochs, model_configs
import argparse


def args_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="pretrained model name")
    return parser.parse_args()


def load_mnist(path, is_train=True):
    """ Load MNIST data from `path`

    :param path: str
    :param is_train: bool
    :return: images: np.array (samples,28,28,1)
    :return:  labels: np.array (samples, 10)

    """


    if is_train:
        prefix = 'train'
    else:
        prefix = 't10k'


    labels_path = os.path.join(path,'{}-labels-idx1-ubyte.gz'.format(prefix))
    images_path = os.path.join(path,'{}-images-idx3-ubyte.gz'.format(prefix))

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 28,28,1)

    images = images/255.0
    labels = keras.utils.to_categorical(labels, 10)
    return images, labels





def plot_img(X, y_true=None, yhat=None, suffix=None):
    """
    plots one image

    :param X: np.array (height, width, 1) of int
    :param y_true: np.array (10,) one hot encoding of the true label
    :param yhat: np.array (10,) one hot encoding of the predicted label
    :param suffix: str
    :return:
    """
    from configs import labels

    X = X.reshape(X.shape[0], X.shape[1]) #convert from 3D to 2D
    y_true = np.argmax(y_true) #convert from one hot encoding to single value label
    yhat = np.argmax(yhat) #convert from one hot encoding to single value label


    fig = plt.figure()
    _ = plt.imshow(X, cmap='gray', vmin=0, vmax=1) #pixels are scaled 0 to 1
    s=""
    if y_true is not None:
        s+= "True value: {}. ".format(labels[int(y_true)])
    if yhat is not None:
        s+="Predicted value: {}".format(labels[int(yhat)])

    plt.title(s)
    fig.savefig('./plots/sample_fig_{}.png'.format(suffix))



def lenet5(batch_norm=False, dropout=None, weight_decay=None, lr=0.01):
    """
    Creates Lenet-5 layer as described here:  https://en.wikipedia.org/wiki/LeNet

    :param batch_norm: bool
    :param dropout:  float
    :param weight_decay: float
    :param lr: float
    :return: keras.Sequential
    """

    if not dropout:
        dropout = 0.0
    if not weight_decay:
        weight_decay = 0.0
    # sequentail API
    model = keras.Sequential()
    # convolutional layer 1
    model.add(keras.layers.InputLayer(input_shape=(28, 28,1)))
    if batch_norm:
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(filters=6,
                                  kernel_size=(5, 5),
                                  strides=(1, 1),
                                  activation='tanh',
                                  padding = 'same',
                                  kernel_regularizer=l2(weight_decay),
                                  kernel_initializer='glorot_uniform'
                                  ))
    # average pooling layer 1
    model.add(keras.layers.AveragePooling2D(pool_size=(2, 2),
                                            strides=(2, 2),
                                            padding='valid'))
    model.add(keras.layers.Dropout(dropout))
    if batch_norm:
        model.add(keras.layers.BatchNormalization())

    # convolutional layer 2
    model.add(keras.layers.Conv2D(filters=16,
                                  kernel_size=(5, 5),
                                  strides=(1, 1),
                                  activation='tanh',
                                  padding='valid',
                                  kernel_regularizer=l2(weight_decay),
                                  kernel_initializer='glorot_uniform'
                                  ))
    # average pooling layer 2
    model.add(keras.layers.AveragePooling2D(pool_size=(2, 2),
                                            strides=(2, 2),
                                            padding='valid'))
    model.add(keras.layers.Dropout(dropout))
    if batch_norm:
        model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())
    # fully connected
    model.add(keras.layers.Dense(units=120,
                                 activation='tanh',
                                 kernel_regularizer=l2(weight_decay),
                                 kernel_initializer='glorot_uniform'
                                 ))
    model.add(keras.layers.Dropout(dropout))
    # fully connected
    model.add(keras.layers.Dense(units=84,
                                 activation='tanh',
                                 kernel_regularizer=l2(weight_decay),
                                 kernel_initializer='glorot_uniform'

                                 ))
    model.add(keras.layers.Dropout(dropout))
    # output layer
    model.add(keras.layers.Dense(units=10,
                                 activation='softmax',
                                 kernel_regularizer=l2(weight_decay),
                                 kernel_initializer='glorot_uniform'
                                 ))

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=lr),
                  metrics=['accuracy'])

    return model

def save_model(model, name):
    model.save('./saved_models/{}.h5'.format(name))

def load_model(name):
    return keras.models.load_model('./saved_models/{}.h5'.format(name))


def plot_results(history, epochs, suffix=None):
    """ Saves plot of convergence graph

    :param history: keras.callbacks.History
    :param epochs: int
    :param suffix: str
    :return:
    """
    num_epochs = np.arange(1,epochs+1)
    plt.figure(dpi=200)
    plt.style.use('ggplot')
    plt.plot(num_epochs, history.history['accuracy'], label='train_acc', c='red')
    plt.plot(num_epochs, history.history['val_accuracy'], label='test_acc', c='green')
    plt.title('Convergence Graph- {}'.format(suffix))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./plots/Convergence Graph- {}.png'.format(suffix))


def reproduce_results(name, X_train, y_train, X_test, y_test):
    """
    Loads pretrained models and reproduces results on test data

    :param name: str name of model to reproduce results
    :return:
    """
    try:
        model = load_model(name)
    except OSError:
        print('Unknown model name : {}'.format(name))
        valid_files = os.listdir('./saved_models')
        valid_names = list(map(lambda x: x.replace('.h5',''),valid_files))
        print('Available names : {}'.format(valid_names))
        exit()

    result_train = model.evaluate(X_train, y_train)
    result_test = model.evaluate(X_test, y_test)
    print(' Train  Loss : {}. Train Acc : {}'.format(result_train[0], result_train[1]))
    print(' Test  Loss : {}. Test Acc : {}'.format(result_test[0], result_test[1]))
    y_hat = model.predict(X_test)
    sampled_indexes = np.random.choice (len(y_hat), size = 5, replace=False)
    print('Plotting 5 random images from test set...')
    for idx in sampled_indexes:
        plot_img(X_test[idx], y_true=y_test[idx], yhat=y_hat[idx], suffix=idx)




if __name__ == '__main__':
    args = args_parsing()
    path = './data'
    print('Loading Data...')
    X_train, y_train = load_mnist(path, is_train=True)
    X_test, y_test = load_mnist(path, is_train=False)
    # Pretrained model
    if args.model:
        print('Trying to load pretraind model : {}'.format(args.model))
        reproduce_results(args.model, X_train, y_train, X_test, y_test)
        exit()
    # Train model
    for config_name in model_configs.keys():
        current_config = model_configs[config_name]
        print('Currently training model {}'.format(config_name))
        print('The configuration is {}'.format(current_config))
        model = lenet5(**current_config)
        model.summary()
        #es = EarlyStopping(monitor='val_loss', mode='max', verbose=1, patience=10)

        history = model.fit(X_train, y_train,
                                   epochs=epochs,
                                   batch_size=batch_size,
                                   verbose=2, validation_data=(X_test, y_test))


        plot_results(history, epochs, suffix=config_name)
        save_model(model, name=config_name)


#TODO make it work on google colab

