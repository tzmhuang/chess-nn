'''
Name: evl_sanity_check.py
Date: 31,Mar,2018
Train on: Mac
Purpose:
        based on evl_NN_2_6
        Increased layer from 3 [0-3] to 6 [0-6]
        With Xaviar initialization
        use only 5000 training sample for sanity Check: should perfect fit

Config:
        Epoch: 5
        batch_size: 10000
        Initialization: rand_normal [mean = 0, std = Xaviar]
        Training: GradientDescentOptimizer
        Step_size: 0.0001
'''

'''
From bucket to terminal:
    gsutil cp gs://chess-nn/test_data.h5 ~/DNN
    gsutil cp gs://chess-nn/train_data.h5 ~/DNN

Get graph/model:
    gcloud compute copy-files nn-instance1:/home/huangtom2/DNN/evl_NN_2_4/ ./
Get model:
    gcloud compute copy-files nn-instance1:/home/huangtom2/DNN/model/... ./
Reset:
    rm ./DNN/graph/evl_NN_2_1/*
'''

import tensorflow as tf
import numpy as np
import random
import h5py


tf.reset_default_graph()

#config
logs_path = "./chess_nn/evl_sanity_check/graph"
saver_dir = "./chess_nn/evl_sanity_check/model/evl_sanity_check"
# logs_path = "./DNN/evl_sanity_check/graph"
# saver_dir = "./DNN/evl_sanity_check/model/evl_sanity_check"



batch_size = 10000
training_epochs = 5

# h = h5py.File("//Volumes/DiskA/chess_data.h5")
train_h = h5py.File("//Volumes/DiskA/train_data.h5")
test_h = h5py.File("//Volumes/DiskA/test_data.h5")

# train_h = h5py.File("./DNN/train_data.h5")
# test_h = h5py.File("./DNN/test_data.h5")

data_size = train_h['flag'].shape[0] + test_h['flag'].shape[0]
partition_train = int(0.7*data_size)
partition_test = data_size - partition_train


def weight_variable(r_num, c_num, name):
    initial = tf.truncated_normal([r_num,c_num], stddev = tf.sqrt(2/r_num))
    return tf.Variable(initial, name = name)

def full_layer(input, W,b):
    return tf.matmul(input, W) + b


'''removed board_set from random batch'''
def rand_batch(h5_ptr,batch_size, data_size):
    pos = random.randint(0, int(data_size/batch_size)-1) * batch_size
    turn_move = h5_ptr['turn_move'][pos:pos+batch_size]
    castling = h5_ptr['castling'][pos:pos+batch_size]
    #board_set = h5_ptr['board_set'][pos:pos+batch_size]
    piece_pos = h5_ptr['piece_pos'][pos:pos+batch_size]
    atk_map = h5_ptr['atk_map'][pos:pos+batch_size]
    flag = h5_ptr['flag'][pos:pos+batch_size]
    current_data = np.concatenate((turn_move,castling,piece_pos,atk_map,flag), axis = 1)
    return current_data


def partition(h5_ptr, data_size,partition_train):
    train = h5py.File('train_data.h5') #./Chess/train_data.h5
    test = h5py.File('test_data.h5')  #./Chess/test_data.h5
    #train
    train_turn_move = h5_ptr['turn_move'][0:partition_train]
    train['turn_move'] = train_turn_move
    train_castling = h5_ptr['castling'][0:partition_train]
    train['castling'] = train_castling
    train_board_set = h5_ptr['board_set'][0:partition_train]
    train['board_set'] = train_board_set #
    train_piece_pos = h5_ptr['piece_pos'][0:partition_train]
    train['piece_pos'] = train_piece_pos
    train_atk_map = h5_ptr['atk_map'][0:partition_train]
    train['atk_map'] = train_atk_map
    train_flag = h5_ptr['flag'][0:partition_train]
    train['flag'] = train_flag
    #test
    test_turn_move = h5_ptr['turn_move'][partition_train:data_size]
    test['turn_move'] = test_turn_move
    test_castling = h5_ptr['castling'][partition_train:data_size]
    test['castling'] = test_castling
    test_board_set = h5_ptr['board_set'][partition_train:data_size]
    test['board_set'] = test_board_set
    test_piece_pos = h5_ptr['piece_pos'][partition_train:data_size]
    test['piece_pos'] = test_piece_pos
    test_atk_map = h5_ptr['atk_map'][partition_train:data_size]
    test['atk_map'] = test_atk_map
    test_flag = h5_ptr['flag'][partition_train:data_size]
    test['flag'] = test_flag
    train.close()
    test.close()
    return

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



global_step = tf.Variable(0, name='global_step',trainable=False)

with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, name = "input", shape = [None,1541])
    y_ = tf.placeholder(tf.float32, name = "flag",shape = [None,3])

tf.summary.histogram('input_x',x)



with tf.name_scope("layer_0"):
    with tf.name_scope("weights_0"):
        W_0 = weight_variable(1541,1541,"W_0")
        variable_summaries(W_0)
    with tf.name_scope("bias_0"):
        b_0 = tf.Variable(initial_value =0.0,name = "b_0")
        variable_summaries(b_0)
    with tf.name_scope("pre_activate_0"):
        a_0 = full_layer(x, W_0, b_0)
        tf.summary.histogram("a_0",a_0)

with tf.name_scope("relu_0"):
    relu_0 = tf.nn.relu(a_0)

tf.summary.histogram("relu_0s",relu_0)


with tf.name_scope("layer_1"):
    with tf.name_scope("weights_1"):
        W_1 = weight_variable(1541,1541,"W_1")
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


with tf.name_scope("layer_2"):
    with tf.name_scope("weights_2"):
        W_2 = weight_variable(1541,1541,"W_2")
        variable_summaries(W_2)
    with tf.name_scope("bias_2"):
        b_2 = tf.Variable(initial_value =0.0,name = "b_2")
        variable_summaries(b_2)
    with tf.name_scope("pre_activate_2"):
        a_2 = full_layer(relu_1, W_2, b_2)
        tf.summary.histogram("a_2",a_2)

with tf.name_scope("relu_2"):
    relu_2 = tf.nn.relu(a_2)

tf.summary.histogram("relu_2s",relu_2)


with tf.name_scope("layer_3"):
    with tf.name_scope("weights_3"):
        W_3 = weight_variable(1541,1541,"W_3")
        variable_summaries(W_3)
    with tf.name_scope("bias_3"):
        b_3 = tf.Variable(initial_value =0.0,name = "b_3")
        variable_summaries(b_3)
    with tf.name_scope("pre_activate_3"):
        a_3 = full_layer(relu_2, W_3, b_3)
        tf.summary.histogram("a_3",a_3)

with tf.name_scope("relu_3"):
    relu_3 = tf.nn.relu(a_3)

tf.summary.histogram("relu_3s",relu_3)



with tf.name_scope("layer_4"):
    with tf.name_scope("weights_4"):
        W_4 = weight_variable(1541,1541,"W_4")
        variable_summaries(W_4)
    with tf.name_scope("bias_4"):
        b_4 = tf.Variable(initial_value =0.0,name = "b_4")
        variable_summaries(b_4)
    with tf.name_scope("pre_activate_4"):
        a_4 = full_layer(relu_3, W_4, b_4)
        tf.summary.histogram("a_4",a_4)

with tf.name_scope("relu_4"):
    relu_4 = tf.nn.relu(a_4)

tf.summary.histogram("relu_4s",relu_4)



with tf.name_scope("layer_5"):
    with tf.name_scope("weights_5"):
        W_5 = weight_variable(1541,1541,"W_5")
        variable_summaries(W_5)
    with tf.name_scope("bias_5"):
        b_5 = tf.Variable(initial_value =0.0,name = "b_5")
        variable_summaries(b_5)
    with tf.name_scope("pre_activate_5"):
        a_5 = full_layer(relu_4, W_5, b_5)
        tf.summary.histogram("a_5",a_5)

with tf.name_scope("relu_5"):
    relu_5 = tf.nn.relu(a_5)

tf.summary.histogram("relu_5s",relu_5)


######output
with tf.name_scope("layer_6"):
    with tf.name_scope("weights_6"):
        W_6 = weight_variable(1541,3,"W_6")
        variable_summaries(W_6)
    with tf.name_scope("bias_6"):
        b_6 = tf.Variable(initial_value =0.0,name = "b_6")
        variable_summaries(b_6)
    with tf.name_scope("pre_activate_6"):
        a_6 = full_layer(relu_5, W_6, b_6)
        tf.summary.histogram("a_6",a_6)

with tf.name_scope("relu_6"):
    relu_6 = tf.nn.relu(a_6)

tf.summary.histogram("relu_6s",relu_6)



with tf.name_scope("prob"):
    Y = tf.nn.softmax( relu_6 )

tf.summary.histogram("softmax", Y)

with tf.name_scope("prediction"):
    pred = tf.argmax(Y, axis = 1)


with tf.name_scope("cross_entropy"):
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(Y+10e-5), reduction_indices=[1]))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits( labels = tf.argmax(y_,1) ,logits = relu_6)
    loss = tf.reduce_mean(cross_entropy)

tf.summary.scalar("loss", loss)


with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss, global_step = global_step)

with tf.name_scope("accuracy"):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, axis = 1), pred),dtype = tf.float32))

tf.summary.scalar("accuracy", accuracy)



#summary output to tensorboard
merged = tf.summary.merge_all()

#model saver
saver = tf.train.Saver()


with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(logs_path + '/train',sess.graph)
    test_writer = tf.summary.FileWriter(logs_path + '/test')
    sess.run(tf.global_variables_initializer())
    test_np = rand_batch(test_h,10000,partition_test)
    test_x = test_np[:,0:1541]  #testing data
    test_y = test_np[:,1541:1544] #testing data
    #writer  =  tf.summary.FileWriter ( logs_path , sess.graph)
    saver.save(sess, saver_dir, global_step=global_step ,write_meta_graph=True)
    batch = rand_batch(train_h,5000,partition_train)
    for epochs in range(training_epochs):
        batch_count = 200
        #print("2")
        for i in range(batch_count):
            train_x = batch[:,0:1541]    #training datas
            train_y = batch[:,1541:1544]   #training flag
            plt_train,cost,train_ac,_ = sess.run([merged,loss,accuracy,train_step], feed_dict={x: train_x,y_: train_y})
            #a,b,c,d,e,f = sess.run([Y_input,y_,tf.log(Y), y_ * tf.log(Y),-tf.reduce_sum(y_ * tf.log(Y), reduction_indices=[1]),tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(Y), reduction_indices=[1]))], feed_dict={x: data_x,y_: data_y})
            #ini,B,W,R = sess.run([tf.truncated_normal([1605,1605], stddev = 0.1),b_0,W_0,x], feed_dict={x: data_x,y_: data_y})
            train_writer.add_summary(plt_train,epochs*batch_count+i) #write log
            # writer.add_summary(plt_test,epochs*batch_count+i)
            if (i)%50 == 0:
                #print("ac_s")
                test_ac,plt_test= sess.run([accuracy,merged], feed_dict={x:test_x, y_: test_y})
                test_writer.add_summary(plt_test,global_step = epochs*batch_count+i)
                #print("ac_f")
                print ("epoch: ",epochs,"iterations: ",i,"cost: ",cost, "train accuracy: ",train_ac, "test accuracy: ",test_ac)
                #c,d = sess.run([tf.log(Y), cross_entropy], feed_dict={x: train_x,y_: train_y})
                #print ("epoch: ",epochs,"iterations: ",i,"cost: ",cost, "train accuracy: ",train_ac)
                saver.save(sess, saver_dir, global_step=global_step ,write_meta_graph=True)
        # if (epochs)%5 == 0:
        #     print ("epochs: ",epochs,"cost: ",cost)
            #print (i,err,p_err)
    #test_cost,l2= sess.run([cost,l2_loss], feed_dict={x: data_x,y_: data_y})
    #test_cost,test_p_cost = sess.run([cost,bias_cost], feed_dict={x: data_x,y_: data_y})
    #model_y, input_y = sess.run([Y, Y_input],feed_dict={x: data_x_500,y_: data_y_500_})
    #print (test_cost,l2)


writer.close()
