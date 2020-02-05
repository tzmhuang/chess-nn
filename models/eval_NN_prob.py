import tensorflow as tf


def weight_variable(r_num, c_num, name):
    initial = tf.truncated_normal([r_num,c_num], stddev = 0.1)
    return tf.Variable(initial, name = name)


def full_layer(input, W,b):
    return tf.matmul(input, W) + b


in_size = int(input.get_shape()[1])


x = tf.placeholder(tf.float32, name = "input", shape = [None,65])
y_ = tf.placeholder(tf.float32, name = "flag",shape = [None,3]) # 3 outputs, prob of win/lose/draw

W_0 = weight_variable(65,65,"W_0")
b_0 = tf.Variable(initial_value =0.0,name = "b_0")
layer_1 = tf.nn.relu(full_layer(x, W_0, b_0))

W_1 = weight_variable(65,65,"W_1")
b_1 = tf.Variable(initial_value= 0.0,name = "b_1")
layer_2 = tf.nn.relu(full_layer(layer_1, W_1, b_1))

W_2 = weight_variable(65,65,"W_2")
b_2 = tf.Variable(initial_value= 0.0,name = "b_2")
layer_3 = tf.nn.relu(full_layer(layer_2, W_2, b_2))

W_3 = weight_variable(65,3,name = "W_3")
b_3 = tf.Variable(initial_value= 0.0,name = "b_3")
Y_input = full_layer(layer_3, W_3, b_3)
Y = tf.nn.softmax( Y_input )


learning_rate = 0.5

#l2_loss = tf.nn.l2_loss(W_0)+tf.nn.l2_loss(W_1)+tf.nn.l2_loss(W_2)+tf.nn.l2_loss(W_3)

cost = tf.reduce_mean(tf.square( Y - y_),name = "cost")#+ 0.01*l2_loss
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(Y), reduction_indices=[1]))
#pure_cost = tf.reduce_mean(tf.square( Y - y_ ))
#bias_cost =tf.reduce_mean(abs( Y - y_ ))
train_step = tf.train.MomentumOptimizer(learning_rate,0.5).minimize(cost)
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

epochs = 5000

correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    writer  =  tf.summary.FileWriter ( "./graph/" , sess.graph)
    sess.run(tf.global_variables_initializer())
    prev_err = 0
    for i in range(epochs):
        err,_,weight_0,weight_1,weight_2,weight_3 = sess.run([cost,train_step,W_0,W_1,W_2,W_3], feed_dict={x: data_x_500,y_: data_y_500_prob})
        #err,p_err,_ = sess.run([cost,pure_cost,train_step], feed_dict={x: data_x_500,y_: data_y_500})
        if i%100 == 0:
            print (i,err)
            #print (i,err,p_err)
        if abs(prev_err - err) < 0.00001:
            break
        prev_err = err
    #test_cost,l2= sess.run([cost,l2_loss], feed_dict={x: data_x,y_: data_y})
    #test_cost,test_p_cost = sess.run([cost,bias_cost], feed_dict={x: data_x,y_: data_y})
    #model_y, input_y = sess.run([Y, Y_input],feed_dict={x: data_x_500,y_: data_y_500_prob})
    #print (test_cost,l2)
    print(sess.run( accuracy, feed_dict={x: data_x_500,y_: data_y_500_prob}))



writer.close()





#tf.reset_default_graph()
