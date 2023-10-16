import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as kl
from keras.losses import binary_crossentropy


def convolution_block(
    x, filters, size, strides=(1, 1), padding="same", activation=True
):
    x = kl.Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = kl.BatchNormalization()(x)
    if activation == True:
        x = kl.Activation("relu")(x)
    return x


def residual_block(blockInput, num_filters=16):
    x = kl.Activation("relu")(blockInput)
    x = kl.BatchNormalization()(x)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = kl.Add()([x, blockInput])
    return x


# Build model
def resnet_model(dim_h, dim_w, start_neurons=16, DropoutRatio=0.5):

    input_layer = kl.Input(shape=(dim_h, dim_w, 1))

    conv1 = kl.Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(
        input_layer
    )
    conv1 = residual_block(conv1, start_neurons * 1)
    conv1 = residual_block(conv1, start_neurons * 1)
    conv1 = kl.Activation("relu")(conv1)
    pool1 = kl.MaxPooling2D((2, 2))(conv1)
    pool1 = kl.Dropout(DropoutRatio / 2)(pool1)

    conv2 = kl.Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2, start_neurons * 2)
    conv2 = residual_block(conv2, start_neurons * 2)
    conv2 = kl.Activation("relu")(conv2)
    pool2 = kl.MaxPooling2D((2, 2))(conv2)
    pool2 = kl.Dropout(DropoutRatio)(pool2)

    conv3 = kl.Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3, start_neurons * 4)
    conv3 = residual_block(conv3, start_neurons * 4)
    conv3 = kl.Activation("relu")(conv3)
    pool3 = kl.MaxPooling2D((2, 2))(conv3)
    pool3 = kl.Dropout(DropoutRatio)(pool3)

    conv4 = kl.Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4, start_neurons * 8)
    conv4 = residual_block(conv4, start_neurons * 8)
    conv4 = kl.Activation("relu")(conv4)
    pool4 = kl.MaxPooling2D((2, 2))(conv4)
    pool4 = kl.Dropout(DropoutRatio)(pool4)

    # Middle
    convm = kl.Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(
        pool4
    )
    convm = residual_block(convm, start_neurons * 16)
    convm = residual_block(convm, start_neurons * 16)
    convm = kl.Activation("relu")(convm)

    deconv4 = kl.Conv2DTranspose(
        start_neurons * 8, (3, 3), strides=(2, 2), padding="same"
    )(convm)
    uconv4 = kl.concatenate([deconv4, conv4])
    uconv4 = kl.Dropout(DropoutRatio)(uconv4)

    uconv4 = kl.Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(
        uconv4
    )
    uconv4 = residual_block(uconv4, start_neurons * 8)
    uconv4 = residual_block(uconv4, start_neurons * 8)
    uconv4 = kl.Activation("relu")(uconv4)

    deconv3 = kl.Conv2DTranspose(
        start_neurons * 4, (3, 3), strides=(2, 2), padding="valid"
    )(uconv4)
    uconv3 = kl.concatenate([deconv3, conv3])
    uconv3 = kl.Dropout(DropoutRatio)(uconv3)

    uconv3 = kl.Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(
        uconv3
    )
    uconv3 = residual_block(uconv3, start_neurons * 4)
    uconv3 = residual_block(uconv3, start_neurons * 4)
    uconv3 = kl.Activation("relu")(uconv3)

    deconv2 = kl.Conv2DTranspose(
        start_neurons * 2, (3, 3), strides=(2, 2), padding="same"
    )(uconv3)
    uconv2 = kl.concatenate([deconv2, conv2])

    uconv2 = kl.Dropout(DropoutRatio)(uconv2)
    uconv2 = kl.Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(
        uconv2
    )
    uconv2 = residual_block(uconv2, start_neurons * 2)
    uconv2 = residual_block(uconv2, start_neurons * 2)
    uconv2 = kl.Activation("relu")(uconv2)

    deconv1 = kl.Conv2DTranspose(
        start_neurons * 1, (3, 3), strides=(2, 2), padding="valid"
    )(uconv2)
    uconv1 = kl.concatenate([deconv1, conv1])

    uconv1 = kl.Dropout(DropoutRatio)(uconv1)
    uconv1 = kl.Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(
        uconv1
    )
    uconv1 = residual_block(uconv1, start_neurons * 1)
    uconv1 = residual_block(uconv1, start_neurons * 1)
    uconv1 = kl.Activation("relu")(uconv1)

    uconv1 = kl.Dropout(DropoutRatio / 2)(uconv1)
    output_layer = kl.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model


@tf.function
def dice_loss(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1, 1])
    y_pred_f = tf.reshape(y_pred, [-1, 1])
    numerator = 2 * tf.reduce_sum(y_true_f * y_pred_f)
    denominator = tf.reduce_sum(y_true_f + y_pred_f)

    return 1 - numerator / denominator


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
