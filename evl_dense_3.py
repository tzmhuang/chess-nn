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
training_epochs = 15

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_dir',"./DNN/evl_dense_3/",'dir of model stored' )
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
    #castling_conv = h5_ptr['castling_conv'][start:fin]
    #castling_conv = castling_conv.reshape((-1,256))
    #board_set = h5_ptr['board_set'][pos:pos+batch_size] #64
    piece_pos = h5_ptr['piece_pos'][start:fin] #768   (12,8,8)
    atk_map = h5_ptr['atk_map'][start:fin] #768
    flag = h5_ptr['flag'][start:fin] #3
    current_data = np.concatenate((turn_move,castling,piece_pos,atk_map,flag), axis = 1)
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
            W_0 = weight_variable(1541,1024,"W_0")
            variable_summaries(W_0)
        with tf.name_scope("bias_0"):
            b_0 = tf.Variable(initial_value =0.0,name = "b_0")
            variable_summaries(b_0)
        with tf.name_scope("pre_activate_0"):
            a_0 = full_layer(input_layer, W_0, b_0)
            tf.summary.histogram("a_0",a_0)
    with tf.name_scope("relu_0"):
        relu_0 = tf.nn.relu(a_0)
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
    with tf.name_scope("relu_1"):
        relu_1 = tf.nn.relu(a_1)
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





def model_fn_mid(features, labels, mode):
    #input
    input_layer = features['x']
    #layer_1
    with tf.name_scope("layer_0"):
        with tf.name_scope("weights_0"):
            W_0 = weight_variable(1541,512,"W_0")
            variable_summaries(W_0)
        with tf.name_scope("bias_0"):
            b_0 = tf.Variable(initial_value =0.0,name = "b_0")
            variable_summaries(b_0)
        with tf.name_scope("pre_activate_0"):
            a_0 = full_layer(input_layer, W_0, b_0)
            tf.summary.histogram("a_0",a_0)
    with tf.name_scope("relu_0"):
        relu_0 = tf.nn.relu(a_0)
    tf.summary.histogram("relu_0s",relu_0)
    dropout_1 = tf.layers.dropout(inputs=relu_0, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN, name = 'Dropout_1')
    #layer_2
    with tf.name_scope("layer_1"):
        with tf.name_scope("weights_1"):
            W_1 = weight_variable(512,256,"W_1")
            variable_summaries(W_1)
        with tf.name_scope("bias_1"):
            b_1 = tf.Variable(initial_value =0.0,name = "b_1")
            variable_summaries(b_1)
        with tf.name_scope("pre_activate_1"):
            a_1 = full_layer(dropout_1, W_1, b_1)
            tf.summary.histogram("a_1",a_1)
    with tf.name_scope("relu_1"):
        relu_1 = tf.nn.relu(a_1)
    tf.summary.histogram("relu_1s",relu_1)
    dropout_2 = tf.layers.dropout(inputs=relu_1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN, name = 'Dropout_2')
    ##logits
    with tf.name_scope("logits"):
        with tf.name_scope("weights_2"):
            W_2 = weight_variable(256,3,"W_2")
            variable_summaries(W_2)
        with tf.name_scope("bias_2"):
            b_2 = tf.Variable(initial_value =0.0,name = "b_2")
            variable_summaries(b_2)
        with tf.name_scope("logit"):
            logit = full_layer(dropout_2, W_2, b_2)
            tf.summary.histogram("logit",logit)
    predictions = {
        'classes': tf.argmax(input=logit, axis=1, name='classes'),
        'probabilities': tf.nn.softmax(logit, name='softmax_tensor')
    }
    tf.summary.histogram('logit',logit)
    tf.summary.histogram('prediction_classes',predictions['classes'])
    tf.summary.histogram('prediction_probabilities',predictions['probabilities'])
    #loss_function
    with tf.name_scope("Loss"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits( labels = tf.argmax(labels,1) ,logits = logit)
        l2_loss = 0.01*(tf.nn.l2_loss(W_0)+tf.nn.l2_loss(W_1)+tf.nn.l2_loss(W_2))
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

move_num_train = train_h['move_num'][:]

train_early = list(np.where(move_num_train<=30)[0])
train_mid = list(np.where((move_num_train>30)*(move_num_train<=60))[0])
train_late = list(np.where(move_num_train>60)[0])

train_early_x = data_1[train_early,0:1541]
train_early_y = data_1[train_early,1541:1544]

train_mid_x = data_1[train_mid,0:1541]
train_mid_y = data_1[train_mid,1541:1544]

train_late_x = data_1[train_late,0:1541]
train_late_y = data_1[train_late,1541:1544]

test_data = h5_get(test_h,0,int(FLAGS.train_data_size)).astype('float32')

move_num_test = test_h['move_num'][:]

test_early = list(np.where(move_num_test<=30)[0])
test_mid = list(np.where((move_num_test>30)*(move_num_test<=60))[0])
test_late = list(np.where(move_num_test>60)[0])

test_early_x = test_data[test_early,0:1541]
test_early_y = test_data[test_early,1541:1544]

test_mid_x = test_data[test_mid,0:1541]
test_mid_y = test_data[test_mid,1541:1544]

test_late_x = test_data[test_late,0:1541]
test_late_y = test_data[test_late,1541:1544]


train_input_fn_early = tf.estimator.inputs.numpy_input_fn(
    x = {'x':train_early_x},
    y = train_early_y,
    batch_size = FLAGS.batch_size,
    num_epochs = 1,
    shuffle = True,
    queue_capacity = 3072
    )

train_input_fn_mid = tf.estimator.inputs.numpy_input_fn(
    x = {'x':train_mid_x},
    y = train_mid_y,
    batch_size = FLAGS.batch_size,
    num_epochs = 1,
    shuffle = True,
    queue_capacity = 3072
    )

train_input_fn_late = tf.estimator.inputs.numpy_input_fn(
    x = {'x':train_late_x},
    y = train_late_y,
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
#EARLY
with tf.device('/gpu:0'):
    evl_dense_early = tf.estimator.Estimator(
        model_fn = model_fn, model_dir = './DNN/evl_dense_3/early/')

#MID
with tf.device('/gpu:0'):
    evl_dense_mid = tf.estimator.Estimator(
        model_fn = model_fn_mid, model_dir ='./DNN/evl_dense_3/mid/')

#LATE
with tf.device('/gpu:0'):
    evl_dense_late = tf.estimator.Estimator(
        model_fn = model_fn_mid, model_dir ='./DNN/evl_dense_3/late/')
#evl_conv_temp.train(
#    input_fn = train_input_fn,hooks = [logging_hook])


early_eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {'x':test_early_x},
    y = test_early_y,
    num_epochs = 1,
    shuffle = False
    )

mid_eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {'x':test_mid_x},
    y = test_mid_y,
    num_epochs = 1,
    shuffle = False
    )

late_eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {'x':test_late_x},
    y = test_late_y,
    num_epochs = 1,
    shuffle = False
    )
#eval_results = evl_conv_temp.evaluate(input_fn=eval_input_fn)

##EARLY
for n in range(FLAGS.epoch):
    print("==================EARLY_eopch{}==================".format(n))
    evl_dense_early.train(
        input_fn = train_input_fn_early,hooks = [logging_hook])
    print('==================EARLY_evaluating==================')
    eval_results = evl_dense_early.evaluate(input_fn=early_eval_input_fn)


##MID
for n in range(FLAGS.epoch):
    print("==================MID_eopch{}==================".format(n))
    evl_dense_mid.train(
        input_fn = train_input_fn_mid,hooks = [logging_hook])
    print('==================MID_evaluating==================')
    eval_results = evl_dense_mid.evaluate(input_fn=mid_eval_input_fn)

##LATE
for n in range(FLAGS.epoch):
    print("==================LATE_eopch{}==================".format(n))
    evl_dense_late.train(
        input_fn = train_input_fn_late,hooks = [logging_hook])
    print('==================LATE_evaluating==================')
    eval_results = evl_dense_late.evaluate(input_fn=late_eval_input_fn)
