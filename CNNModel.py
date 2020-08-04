import tensorflow as tf
import efficientnet.keras as efn
from keras.models import Model
from keras.layers import Lambda
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization, Input

# Generalized mean pool - GeM
gm_exp = tf.Variable(3.0, dtype=tf.float32)


def generalized_mean_pool_2d(X):
    pool = (tf.reduce_mean(tf.abs(X ** (gm_exp)),
                           axis=[1, 2],
                           keepdims=False) + 1.e-7) ** (1. / gm_exp)
    return pool


def get_model(input_shape):

    # Input Layer
    input = Input(shape=input_shape)

    x_model = efn.EfficientNetB3(weights='imagenet', include_top=False, input_tensor=input,
                                 pooling=None, classes=None)

    # UnFreeze all layers
    for layer in x_model.layers:
        layer.trainable = True

    lambda_layer = Lambda(generalized_mean_pool_2d)
    lambda_layer.trainable_weights.extend([gm_exp])
    x = lambda_layer(x_model.output)

    # multi output
    season = Dense(4, activation="sigmoid", name="season")(x)

    # model
    model = Model(inputs=x_model.input, outputs=[season])

    return model
