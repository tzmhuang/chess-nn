'''
Name: evl_conv_temp
Date: 23,Apr,2018
Train on: Google VM, 4 CPU + NVIDIA K80 GPU
Purpose:
        - Adamoptimizer
        - using batch size: 512
        - using more random sample, shuffle -> iter from start
            - failed due to memory error
            - retry with another implementation
        - Using one_hot representation
        - Adding move_num as training input, hopefully help machine distinguish stages of game
        - using uniform Xaviar initizlization
Config:
        - Epoch: 5
        - batch_size: 1024
        - Training: Adamoptimizer
        - learning rate: 0.001
        - beta 0.01
        - beta1 = 0.9
        - beta2 = 0.999
'''

'''
From bucket to terminal:
    gsutil cp gs://chess-nn/test_data.h5 ~/DNN
    gsutil cp gs://chess-nn/train_data.h5 ~/DNN

Get graph/model:
    gcloud compute copy-files nn-instance1:/home/huangtom2/DNN/evl_conv_temp/ ./
Get model:
    gcloud compute copy-files nn-instance1:/home/huangtom2/DNN/model/... ./
Reset:
    rm -r ./DNN/evl_conv_temp
'''


import tensorflow as tf
import numpy as np
import random
import h5py


tf.reset_default_graph()

train_h = h5py.File("./DNN/train_data.h5")
test_h = h5py.File("./DNN/test_data.h5")

data_size = train_h['flag'].shape[0] + test_h['flag'].shape[0]
partition_train = int(0.7*data_size)
partition_test = data_size - partition_train

batch_size = 512
training_epochs = 5

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_dir',"./DNN/evl_conv_temp/",'dir of model stored' )
tf.app.flags.DEFINE_integer('train_data_size',partition_train, 'size of training data')
tf.app.flags.DEFINE_integer('test_data_size',partition_test, 'size of testing data')
tf.app.flags.DEFINE_integer('batch_size',batch_size, 'mini batch size' )
tf.app.flags.DEFINE_integer('epoch',5, 'total epoch trained')



def h5_get(h5_ptr,start,fin):
    #move_num = h5_ptr['move_num'][pos:pos+batch_size] #1
    #game_phase = h5_ptr['game_phase'][ind] #3
    #turn_move = h5_ptr['turn_move'][ind] #1
    turn_move_conv = h5_ptr['turn_move_conv'][start:fin]
    #castling = h5_ptr['castling'][ind] #4
    castling_conv = h5_ptr['castling_conv'][start:fin]
    castling_conv = castling_conv.reshape((-1,256))
    #board_set = h5_ptr['board_set'][pos:pos+batch_size] #64
    piece_pos = h5_ptr['piece_pos'][start:fin] #768   (12,8,8)
    #atk_map = h5_ptr['atk_map'][ind] #768
    flag = h5_ptr['flag'][start:fin] #3
    current_data = np.concatenate((piece_pos,turn_move_conv,castling_conv,flag), axis = 1)
    del(turn_move_conv,castling_conv,piece_pos)
    return current_data


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)



def model_fn(features, labels, mode):
    #input and reshape
    input_layer = tf.reshape(features['x'],[-1,8,8,17])
    #conv1
    conv1 = tf.layers.conv2d(inputs=input_layer,filters=64,kernel_size=[3, 3],
            padding="same",activation=None,name='CONV1')
    relu1 = tf.nn.relu(conv1, name = 'relu1')
    tf.summary.histogram('conv1', conv1)
    tf.summary.histogram('relu1', relu1)
    #bn1 = tf.layers.batch_normalization(input = conv1,training=mode == tf.estimator.ModeKeys.TRAIN,name='BN1')
    #conv2
    conv2 = tf.layers.conv2d(inputs=relu1, filters=64, kernel_size=[3, 3],
            padding="same", activation=tf.nn.relu,name='CONV2')
    relu2 = tf.nn.relu(conv2, name = 'relu2')
    tf.summary.histogram('conv2', conv2)
    tf.summary.histogram('relu2', relu2)
    #conv3
    conv3 = tf.layers.conv2d(inputs=relu2, filters=64, kernel_size=[3, 3],
            padding="same", activation=tf.nn.relu,name='CONV3')
    relu3 = tf.nn.relu(conv3, name = 'relu3')
    tf.summary.histogram('conv3', conv3)
    tf.summary.histogram('relu3', relu3)
    #dense_layer
    flattern = tf.reshape(relu3, [-1,8*8*64])
    dense_1 = tf.layers.dense(inputs=flattern, units=512, name='DENSE_1')
    #tf.summary.histogram('flat', flattern)
    logit = tf.layers.dense(inputs=dense_1, units=3, name='FINAL')
    # dropout = tf.layers.dropout(
    #   inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    predictions = {
        'classes': tf.argmax(input=logit, axis=1, name='classes'),
        'probabilities': tf.nn.softmax(logit, name='softmax_tensor')
    }
    tf.summary.histogram('logit',logit)
    tf.summary.histogram('prediction_classes',predictions['classes'])
    tf.summary.histogram('prediction_probabilities',predictions['probabilities'])
    #loss_function
    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits( labels = tf.argmax(labels,1) ,logits = logit)
        loss = tf.reduce_mean(cross_entropy, name = 'mean_loss')
    tf.summary.scalar('training_loss', loss)
    #Assess Accuracy
    accuracy, update_op = tf.metrics.accuracy(
        labels=tf.argmax(labels, axis = 1), predictions=predictions['classes'], name='accuracy')
    batch_acc = tf.reduce_mean(tf.cast(
        tf.equal(tf.cast(tf.argmax(labels, axis = 1), tf.int64), predictions['classes']), tf.float32))
    tf.summary.scalar('batch_acc', batch_acc)
    tf.summary.scalar('streaming_acc', update_op)
    #training_operation
    if mode == tf.estimator.ModeKeys.TRAIN:
        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    eval_metric_ops = {
        'accuracy': (accuracy, update_op)
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



# def train_input_fn():
#     data_x = h5_get(test_h,0,)[0:768]
#     data_y = h5_get(test_h,0,100)[:,768:771]
#     dataset = tf.data.Dataset.from_tensor_slices(({'x':data_x},data_y ))
#     dataset = dataset.shuffle(256).repeat(FLAG.epoch).batch(FLAG.batch_size)
#     return dataset.make_one_shot_iterator().get_next()

data_1 = h5_get(train_h,0,int(FLAGS.train_data_size/2)).astype('float32')
data_2 = h5_get(train_h,int(FLAGS.train_data_size/2),FLAGS.train_data_size).astype('float32')
data_1 = np.concatenate((data_1,data_2),axis = 0)
del(data_2)

data_x = data_1[:,0:1088]
data_y = data_1[:,1088:1091]

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {'x':data_x},
    y = data_y,
    batch_size = FLAGS.batch_size,
    num_epochs = FLAGS.epoch,
    shuffle = True,
    queue_capacity = 1024
    )

#logging for prediction and training
tensors_to_log = {'probabilities':'softmax_tensor'}
logging_hook = tf.train.LoggingTensorHook(
    tensors = tensors_to_log, every_n_secs = 300)

#training on gpu
with tf.device('/gpu:0'):
    evl_conv_temp = tf.estimator.Estimator(
        model_fn = model_fn, model_dir = FLAGS.model_dir)

evl_conv_temp.train(
    input_fn = train_input_fn,hooks = [logging_hook])


test_data = h5_get(test_h,0,int(FLAGS.train_data_size)).astype('float32')
test_x = test_data[:,0:1088]
test_y = test_data[:,1088:1091]

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {'x':test_x},
    y = test_y,
    num_epochs = 1,
    shuffle = False
    )

eval_results = evl_conv_temp.evaluate(input_fn=eval_input_fn)
