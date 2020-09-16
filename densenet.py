from  keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K

def conv_factory(net, concat_axis, nb_filter,
                 dropout_rate=None):
    """
    :param net: Input keras network
    :param concat_axis: concate axis
    :param nb_filter: number of filters
    :param dropout_rate: you know what it is
    :return: keras network
    """

    net = BatchNormalization(axis=concat_axis)(net)
    net = Activation('relu')(net)
    net = Conv2D(nb_filter,(3, 3), kernel_initializer='he_uniform',padding='same')
    if dropout_rate:
        net = Dropout(dropout_rate)(net)

    return net


def transition(net, concat_axis, nb_filters, dropout_rate=None):

    """

    :param net: keras model
    :param concat_axis: axis
    :param nb_filters: n of filter
    :param dropout_rate: rate
    :return: model
    """
    net = BatchNormalization(axis=concat_axis)(net)
    net = Activation('relu')(net)
    net = Conv2D(nb_filters, (1, 1), padding='same',
               kernel_initializer='he_uniform')(net)
    if dropout_rate:
        net = Dropout(dropout_rate)(net)

    net = AveragePooling2D((2, 2), strides=(2, 2))(net)

    return net


def dense_block(net, concat_axis, nb_layers, nb_filters, growth, dropout_rate=None):

    list_features = [net]

    for i in range(nb_layers):
        net = conv_factory(net, concat_axis, growth, dropout_rate)
        list_features.append(net)
        net = Concatenate(axis=concat_axis)(list_features)

    return net, nb_filters

def DenseNet(nb_classes, img_dim, depth, nb_dense_block, growth_rate,
             nb_filter, dropout_rate=None):


    if K.image_dim_ordering() == "th":
        concat_axis = 1

    elif K.image_dim_ordering() == "tf":
        concat_axis = -1

    model_input = Input(shape=img_dim)

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = Conv2D(nb_filter, (3, 3),
               kernel_initializer="he_uniform",
               padding="same",
               name="initial_conv2D",
               use_bias=False)
    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, concat_axis, nb_layers,
                                  nb_filter, growth_rate,
                                   dropout_rate=dropout_rate)
        # add transition
        x = transition(x,concat_axis, nb_filter, dropout_rate=dropout_rate)

    # The last denseblock does not have a transition
    x, nb_filter = dense_block(x, concat_axis, nb_layers,
                              nb_filter, growth_rate,
                              dropout_rate=dropout_rate)

    x = BatchNormalization(axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D(data_format=K.image_data_format())(x)
    x = Dense(nb_classes,
              activation='softmax')(x)

    densenet = Model(inputs=[model_input], outputs=[x], name="DenseNet")

    return densenet

