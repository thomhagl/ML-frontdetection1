import numpy as np
from numpy.lib.function_base import gradient
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.gen_array_ops import tensor_scatter_add

model_input = keras.Input(shape=(100, 100, 1))

"-- Main blocks level --"
block1_1 = layers.Conv2D(filters=64, kernel_size=(3,3), padding="same")(model_input)
block1_2 = layers.Conv2D(filters=64, kernel_size=(3,3), padding="same")(block1_1)
#batchnormal1 = layers.BatchNormalization()(block1_2)
block1_maxpool = layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=1)(block1_2)

block2_1 = layers.Conv2D(filters=128, kernel_size=(3,3), padding="same")(block1_maxpool)
block2_2 = layers.Conv2D(filters=128, kernel_size=(3,3), padding="same")(block2_1)
#batchnormal2 = layers.BatchNormalization()(block2_2)
block2_maxpool = layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=1)(block2_2)

block3_1 = layers.Conv2D(filters=256, kernel_size=(3,3), padding="same")(block2_maxpool)
block3_2 = layers.Conv2D(filters=256, kernel_size=(3,3), padding="same")(block3_1)
block3_3 = layers.Conv2D(filters=256, kernel_size=(3,3), padding="same")(block3_2)
#batchnormal3 = layers.BatchNormalization()(block3_3)
block3_maxpool = layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=1)(block3_3)

block4_1 = layers.Conv2D(filters=512, kernel_size=(3,3), padding="same")(block3_maxpool)
block4_2 = layers.Conv2D(filters=512, kernel_size=(3,3), padding="same")(block4_1)
block4_3 = layers.Conv2D(filters=512, kernel_size=(3,3), padding="same")(block4_2)

"-- Sub-blocks level --"
subblock1_1 = layers.Conv2D(filters=21, kernel_size=(1,1), padding="same")(block1_1)
subblock1_2 = layers.Conv2D(filters=21, kernel_size=(1,1), padding="same")(block1_2)

subblock2_1 = layers.Conv2D(filters=21, kernel_size=(1,1), padding="same")(block2_1)
subblock2_2 = layers.Conv2D(filters=21, kernel_size=(1,1), padding="same")(block2_2)

subblock3_1 = layers.Conv2D(filters=21, kernel_size=(1,1), padding="same")(block3_1)
subblock3_2 = layers.Conv2D(filters=21, kernel_size=(1,1), padding="same")(block3_2)
subblock3_3 = layers.Conv2D(filters=21, kernel_size=(1,1), padding="same")(block3_3)

subblock4_1 = layers.Conv2D(filters=21, kernel_size=(1,1), padding="same")(block4_1)
subblock4_2 = layers.Conv2D(filters=21, kernel_size=(1,1), padding="same")(block4_2)
subblock4_3 = layers.Conv2D(filters=21, kernel_size=(1,1), padding="same")(block4_3)

" -- Summing level -- "
output_sum1 = layers.Add()([subblock1_1, subblock1_2])
subsubblock1 = layers.Conv2D(filters=1, kernel_size=(1,1), padding="same")(output_sum1)

output_sum2 = layers.Add()([subblock2_1, subblock2_2])
subsubblock2 = layers.Conv2D(filters=1, kernel_size=(1,1), padding="same")(output_sum2)
deconv2 = layers.Conv2DTranspose(filters=1, kernel_size=(1,1))(subsubblock2)

output_sum3 = layers.Add()([subblock3_1, subblock3_2, subblock3_3])
subsubblock3 = layers.Conv2D(filters=1, kernel_size=(1,1), padding="same")(output_sum3)
deconv3 = layers.Conv2DTranspose(filters=1, kernel_size=(1,1))(subsubblock3)

output_sum4 = layers.Add()([subblock4_1, subblock4_2, subblock4_3])
subsubblock4 = layers.Conv2D(filters=1, kernel_size=(1,1), padding="same")(output_sum4)
deconv4 = layers.Conv2DTranspose(filters=1, kernel_size=(1,1))(subsubblock4)

"-- Final layer --"
concatblock = layers.Concatenate()([subsubblock1, deconv2, deconv3, deconv4])
model_output = layers.Conv2D(filters=1, kernel_size=(1,1), padding="same")(concatblock)

model = keras.Model(inputs=model_input, outputs=model_output)

def sigmoid(x):

    activation = 1/(1+tf.math.exp(-x))

    return activation


def loss_wce(x, r, beta, front_indicator):

    if front_indicator:
        loss = -beta*tf.math.log(sigmoid(x))
    else:
        loss = -r*(1-beta)*tf.math.log(1-sigmoid(x))
    
    return tf.cast(loss, tf.float32)

def loss_iou(output_target, output_predicted):

    shape = keras.backend.shape(output_predicted)

    predicted_intersection = tf.math.count_nonzero(output_target*output_predicted)
    predicted_union = tf.math.count_nonzero(output_target + output_predicted)

    iou = predicted_intersection/predicted_union

    l_iou = 1 - iou

    return tf.cast(l_iou, tf.float32)

def custom_loss(y_true, y_pred):
    return (y_true-y_pred)

count = 1

def wein_loss(X, batch_number, alpha, r):

    X = tf.reshape(X,shape=(batch_number,100,100,1))

    def loss(y_true, y_pred):

        global count

        shape = keras.backend.shape(y_pred)

        if count > batch_number:
            count = 1

        #output1 = model.get_layer(index=31)(X[count-1:count,:,:,:])
        output2 = model.get_layer(index=32)(X[count-1:count,:,:,:])
        output3 = model.get_layer(index=33)(X[count-1:count,:,:,:])
        output4 = model.get_layer(index=34)(X[count-1:count,:,:,:])

        block_output = [output2, output3, output4]
        
        Y_front = tf.math.count_nonzero(y_true)
        Y_nonfront = tf.cast(shape[1]*shape[2], tf.int64)  - tf.cast(Y_front, tf.int64)

        beta = tf.cast(Y_nonfront/(Y_front + Y_nonfront), tf.float32)

        L = 0.0

        for i in range(shape[1].numpy()):
            for j in range(shape[0].numpy()):
                for output in block_output:
                
                    l1 = loss_wce(output[0,i,j,0], r, beta, 1) + alpha*loss_iou(y_true, y_pred)

                    L = L + l1
                
                l2 = loss_wce(y_pred[0,i,j], r, beta, y_true[0,i,j].numpy()) + alpha*loss_iou(y_true, y_pred)
                L = L + l2

        count = count + 1

        return L

    return loss


def iou(y_true, y_pred):

    predicted_intersection = tf.math.count_nonzero(y_true*y_pred)
    predicted_union = tf.math.count_nonzero(y_true + y_pred)

    iou = predicted_intersection/predicted_union

    return tf.cast(iou, tf.float32)

X_data = np.zeros((21,100,100))
Y_data = np.zeros((21,100,100))

for k in range(0, 21):

    filename = "example" + str(k) + ".csv"

    x = np.loadtxt("./cropped_examples/input/" + filename)
    y = np.loadtxt("./cropped_examples/output/" + filename)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i,j] < 0:
                x[i,j] = np.nan
    
    grad = np.gradient(x)

    x = np.nan_to_num(np.sqrt(np.square(grad[0]) + np.square(grad[1])))
    y = np.nan_to_num(y)
    
    for i in range(0,100):
        for j in range(0,100):
            X_data[k, i, j] = x[i,j]
            Y_data[k, i, j] = y[i,j]

x_train = X_data[0:19, :, :]
x_test = X_data[20:21, :, :]
y_train = Y_data[0:19, :, :]
y_test = Y_data[20:21, :, :]

model.compile(
    loss=wein_loss(x_train, 19, 1000, 2),
    optimizer=keras.optimizers.SGD(learning_rate=1e-6, momentum=0.1),
    metrics=iou,
    run_eagerly=True,
)

history = model.fit(x_train, y_train, epochs=50, batch_size=1, shuffle=False)

test_scores = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

model.save_weights('my_model_weights8.h5')


