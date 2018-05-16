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

batch_size = 1024
training_epochs =20

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_dir',"./DNN/evl_dense_4/",'dir of model stored' )
tf.app.flags.DEFINE_integer('train_data_size',partition_train, 'size of training data')
tf.app.flags.DEFINE_integer('test_data_size',partition_test, 'size of testing data')
tf.app.flags.DEFINE_integer('batch_size',batch_size, 'mini batch size' )
tf.app.flags.DEFINE_integer('epoch',training_epochs, 'total epoch trained')



def h5_get(h5_ptr,start,fin):
    #move_num = h5_ptr['move_num'][pos:pos+batch_size] #1
    #game_phase = h5_ptr['game_phase'][ind] #3
    turn_move = h5_ptr['turn_move'][start:fin] #1
    #turn_move_conv = h5_ptr['turn_move_conv'][start:fin]
    castling = h5_ptr['castling'][start:fin] #4
    board_check = h5_ptr['board_check'][start:fin]#2
    piece_num = h5_ptr['piece_num'][start:fin] #12
    #castling_conv = h5_ptr['castling_conv'][start:fin]
    #castling_conv = castling_conv.reshape((-1,256))
    #board_set = h5_ptr['board_set'][start:fin] #64
    #board_set = board_set/10
    piece_pos = h5_ptr['piece_pos'][start:fin] #768   (12,8,8)
    atk_map = h5_ptr['atk_map'][start:fin] #768
    flag = h5_ptr['flag'][start:fin] #3
    current_data = np.concatenate((turn_move,castling,board_check,piece_num,piece_pos,atk_map,flag), axis = 1)
    del(turn_move,castling,piece_pos,atk_map)
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

def weight_variable(r_num, c_num, name):
    initial = tf.truncated_normal([r_num,c_num], stddev = tf.sqrt(2/r_num))
    return tf.Variable(initial, name = name)

def full_layer(input, W,b):
    return tf.matmul(input, W) + b


def model_fn(features, labels, mode):
    #input
    input_layer = features['x']
    #layer_1
    with tf.name_scope("layer_0"):
        with tf.name_scope("weights_0"):
            W_0 = weight_variable(1555,1024,"W_0")
            variable_summaries(W_0)
        with tf.name_scope("bias_0"):
            b_0 = tf.Variable(initial_value =0.0,name = "b_0")
            variable_summaries(b_0)
        with tf.name_scope("pre_activate_0"):
            a_0 = full_layer(input_layer, W_0, b_0)
            tf.summary.histogram("a_0",a_0)
    bn0 = tf.layers.batch_normalization(
            inputs = a_0, training= mode==tf.estimator.ModeKeys.TRAIN, name = 'BN0')
    with tf.name_scope("relu_0"):
        relu_0 = tf.nn.relu(bn0)
    tf.summary.histogram("relu_0s",relu_0)
    #layer_2
    with tf.name_scope("layer_1"):
        with tf.name_scope("weights_1"):
            W_1 = weight_variable(1024,512,"W_1")
            variable_summaries(W_1)
        with tf.name_scope("bias_1"):
            b_1 = tf.Variable(initial_value =0.0,name = "b_1")
            variable_summaries(b_1)
        with tf.name_scope("pre_activate_1"):
            a_1 = full_layer(relu_0, W_1, b_1)
            tf.summary.histogram("a_1",a_1)
    bn1 = tf.layers.batch_normalization(
            inputs = a_1, training= mode==tf.estimator.ModeKeys.TRAIN, name = 'BN1')
    with tf.name_scope("relu_1"):
        relu_1 = tf.nn.relu(bn1)
    tf.summary.histogram("relu_1s",relu_1)
    dropout = tf.layers.dropout(inputs=relu_1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN, name = 'Dropout')
    ##logits
    with tf.name_scope("logits"):
        with tf.name_scope("weights_2"):
            W_2 = weight_variable(512,3,"W_2")
            variable_summaries(W_2)
        with tf.name_scope("bias_2"):
            b_2 = tf.Variable(initial_value =0.0,name = "b_2")
            variable_summaries(b_2)
        with tf.name_scope("logit"):
            logit = full_layer(dropout, W_2, b_2)
            tf.summary.histogram("logit",logit)
    predictions = {
        'classes': tf.argmax(input=logit, axis=1, name='classes'),
        'probabilities': tf.nn.softmax(logit, name='softmax_tensor')
    }
    tf.summary.histogram('logit',logit)
    tf.summary.histogram('prediction_classes',predictions['classes'])
    tf.summary.histogram('prediction_probabilities',predictions['probabilities'])
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)
    #loss_function
    with tf.name_scope("Loss"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits( labels = tf.argmax(labels,1) ,logits = logit)
        l2_loss = 0.001*(tf.nn.l2_loss(W_0)+tf.nn.l2_loss(W_1)+tf.nn.l2_loss(W_2))
        loss = tf.reduce_mean(cross_entropy, name = 'mean_loss') + l2_loss
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
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
        with tf.control_dependencies(update_ops):
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

data_x = data_1[:,0:1555]
data_y = data_1[:,1555:1558]

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {'x':data_x},
    y = data_y,
    batch_size = FLAGS.batch_size,
    num_epochs = 1,
    shuffle = True,
    queue_capacity = 3072
    )

#logging for prediction and training
tensors_to_log = {'probabilities':'softmax_tensor'}
logging_hook = tf.train.LoggingTensorHook(
    tensors = tensors_to_log, every_n_secs = 60)

#training on gpu
with tf.device('/gpu:0'):
    evl_conv_temp = tf.estimator.Estimator(
        model_fn = model_fn, model_dir =FLAGS.model_dir)

#evl_conv_temp.train(
#    input_fn = train_input_fn,hooks = [logging_hook])


test_data = h5_get(test_h,0,int(FLAGS.train_data_size/5)).astype('float32')
test_x = test_data[:,0:1555]
test_y = test_data[:,1555:1558]

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {'x':test_x},
    y = test_y,
    num_epochs = 1,
    shuffle = False
    )

#eval_results = evl_conv_temp.evaluate(input_fn=eval_input_fn)


for n in range(FLAGS.epoch):
    print("==================eopch{}==================".format(n))
    evl_conv_temp.train(
        input_fn = train_input_fn,hooks = [logging_hook])
    print('==================evaluating==================')
    eval_results = evl_conv_temp.evaluate(input_fn=eval_input_fn)
