from Desktop.chess-nn.minimax_lookup import *
from Desktop.chess-nn.evl_conv_3 import model_fn
import time
meta_dir = "./DNN/evl_conv_final4/model.ckpt-106481.meta"#select middle ckpt
#meta_dir = "./DNN/evl_NN_Adam/model/evl_NN_Adam-23129.meta"


tf.reset_default_graph()

imported_meta = tf.train.import_meta_graph(meta_dir)


'''Initiating Variables...'''
sess = tf.Session()
imported_meta.restore(sess, tf.train.latest_checkpoint('./DNN/evl_conv_final4/'))
# imported_meta.restore(sess, tf.train.latest_checkpoint('./DNN/evl_NN_Adam/model/'))
graph = tf.get_default_graph()
kernel_1 = graph.get_tensor_by_name("CONV1/kernel:0")
bias_1 = graph.get_tensor_by_name("CONV1/bias:0")
kernel_2 = graph.get_tensor_by_name("CONV2/kernel:0")
bias_2 = graph.get_tensor_by_name("CONV2/bias:0")
kernel_3 = graph.get_tensor_by_name("CONV3/kernel:0")
bias_3 = graph.get_tensor_by_name("CONV3/bias:0")
# kernel_4 = graph.get_tensor_by_name("CONV4/kernel:0")
# bias_4 = graph.get_tensor_by_name("CONV4/bias:0")
# kernel_5 = graph.get_tensor_by_name("CONV5/kernel:0")
# bias_5 = graph.get_tensor_by_name("CONV5/bias:0")

W = graph.get_tensor_by_name("DENSE_1/kernel:0")
b = graph.get_tensor_by_name("DENSE_1/bias:0")
final_W = graph.get_tensor_by_name("FINAL/kernel:0")
final_b = graph.get_tensor_by_name("FINAL/bias:0")

bn1_mean = graph.get_tensor_by_name('BN1/moving_mean:0')
bn2_mean = graph.get_tensor_by_name('BN2/moving_mean:0')
bn3_mean = graph.get_tensor_by_name('BN3/moving_mean:0')
# bn4_mean = graph.get_tensor_by_name('BN4/moving_mean:0')
# bn5_mean = graph.get_tensor_by_name('BN5/moving_mean:0')
bn1_var = graph.get_tensor_by_name('BN1/moving_variance:0')
bn2_var = graph.get_tensor_by_name('BN2/moving_variance:0')
bn3_var = graph.get_tensor_by_name('BN3/moving_variance:0')
# bn4_var = graph.get_tensor_by_name('BN4/moving_variance:0')
# bn5_var = graph.get_tensor_by_name('BN5/moving_variance:0')
bn1_gamma = graph.get_tensor_by_name('BN1/gamma:0')
bn2_gamma = graph.get_tensor_by_name('BN2/gamma:0')
bn3_gamma = graph.get_tensor_by_name('BN3/gamma:0')
# bn4_gamma = graph.get_tensor_by_name('BN4/gamma:0')
# bn5_gamma = graph.get_tensor_by_name('BN5/gamma:0')
bn1_beta = graph.get_tensor_by_name('BN1/beta:0')
bn2_beta = graph.get_tensor_by_name('BN2/beta:0')
bn3_beta = graph.get_tensor_by_name('BN3/beta:0')
# bn4_beta = graph.get_tensor_by_name('BN4/beta:0')
# bn5_beta = graph.get_tensor_by_name('BN5/beta:0')



x = tf.placeholder(tf.float32,name = 'input', shape = [None,8,8,43] )

conv_1 = tf.nn.conv2d(x,kernel_1,[1,1,1,1],'SAME', name = 'conv_1')
conv_1_out = tf.nn.bias_add(conv_1,bias_1, data_format = 'NHWC',name = 'conv_1_out')
bn1 = tf.nn.batch_normalization(conv_1_out, mean =bn1_mean ,variance = bn1_var,
        offset =bn1_beta, scale =bn1_gamma, variance_epsilon = 10e-7 )
relu1 = tf.nn.relu(bn1)

conv_2 = tf.nn.conv2d(relu1,kernel_2,[1,1,1,1],'SAME',name = 'conv_2')
conv_2_out = tf.nn.bias_add(conv_2,bias_2, data_format = 'NHWC',name = 'conv_2_out')
bn2 = tf.nn.batch_normalization(conv_2_out, mean =bn2_mean ,variance = bn2_var,
        offset =bn2_beta, scale =bn2_gamma, variance_epsilon = 10e-7 )
relu2 = tf.nn.relu(bn2)

conv_3 = tf.nn.conv2d(relu2,kernel_3,[1,1,1,1],'SAME',name = 'conv_3')
conv_3_out = tf.nn.bias_add(conv_3,bias_3, data_format = 'NHWC',name = 'conv_3_out')
bn3 = tf.nn.batch_normalization(conv_3_out, mean =bn3_mean ,variance = bn3_var,
        offset =bn3_beta, scale =bn3_gamma, variance_epsilon = 10e-7 )
relu3 = tf.nn.relu(bn3)

# conv_4 = tf.nn.conv2d(relu3,kernel_4,[1,1,1,1],'SAME')
# conv_4_out = tf.nn.bias_add(conv_4,bias_4, data_format = 'NHWC')
# bn4 = tf.nn.batch_normalization(conv_4_out, mean =bn4_mean ,variance = bn4_var,
#         offset =bn4_beta, scale =bn4_gamma, variance_epsilon = 10e-7 )
# relu4 = tf.nn.relu(bn4)
#
# conv_5 = tf.nn.conv2d(relu4,kernel_5,[1,1,1,1],'SAME')
# conv_5_out = tf.nn.bias_add(conv_5,bias_5, data_format = 'NHWC')
# bn5 = tf.nn.batch_normalization(conv_5_out, mean =bn5_mean ,variance = bn5_var,
#         offset =bn5_beta, scale =bn5_gamma, variance_epsilon = 10e-7 )
# relu5 = tf.nn.relu(bn5)

flattern = tf.reshape(relu3,[-1,8*8])
dense = tf.matmul(flattern,W) + b
relu_final = tf.nn.relu(dense)

logit_layer = tf.matmul(relu_final,final_W) + final_b
prediction = tf.nn.softmax(logit_layer)


player_side = input("Please choose your side(b/w): ")

difficulty = input("Choose Difficulty(1-10): ")
board = chess.Board()
#done? implement a-b pruning for memory saving
def minimax_lookup(board, depth, alpha, beta, max_state):
    if leaf(board) or depth == 0:
        in_board = extract_conv(board)
        #print("evl")
        v = evaluate(in_board)
        #print("leaf")
        #print(v)
        #print("evl_fin")
    else:
        legal_moves = board.legal_moves
        child_v = np.empty([1,0])
        if max_state:
            #print("max node")
            v = float('-inf')
            for moves in legal_moves:
                child_board = board.copy()
                child_board.push(moves)
                tmp = minimax_lookup(child_board,depth-1,v,beta, False)
                v = max(tmp,v)
                if v >= beta:
                    #print("max prune")
                    return beta
        if not max_state:
            #print("min node")
            v = float('inf')
            for moves in legal_moves:
                child_board = board.copy()
                child_board.push(moves)
                tmp = minimax_lookup(child_board,depth-1,alpha,v ,True)
                v = min(v,tmp)
                if v <= alpha:
                    print("min prune")
                    return alpha
    return v

#need to consider lose or winning senarios
def evaluate(in_data):
    #start_time = time.time()
    Y = sess.run(prediction,feed_dict={x:in_data})
    #print (time.time()-start_time)
    return Y[0][0] #winning
    #return time.time-start_time

def get_val(board):
    in_data = extract_conv(board)
    graph = tf.get_default_graph()
    Y = graph.get_tensor_by_name("prob/Softmax:0")
    epsilon = tf.constant(0.00000000001)
    v = sess.run(Y, feed_dict={x:in_data})
    return v


# Number of moves from board_a to board_b
def nxtmv_count(data_np,board_a, board_b):
    ind = np.where((data_np == board_a).sum(axis = 1) == 64)
    ind = ind+1
    temp = data_np[ind]
    count = ((temp == board_b).sum(axis = 1)== 64).sum()
    return count

def next_move(board):
    result = np.empty([0,64], dtype=int)
    for move in board.legal_moves:
        tmp_board = board.copy()
        tmp_board.push(move)
        tmp = board_cvrt(tmp_board)
        result = np.concatenate((result,tmp), axis = 0)
    return result

def drichlet_count(data_np,board):
    count = np.empty([0,1])
    board_a = board_cvrt(board).reshape([64])
    legal_move = next_move(board)
    for board_b in legal_move:
        tmp = nxtmv_count(data_np,board_a,board_b)
        count = np.concatenate((count,np.array([[tmp]])), axis = 0)
    return count


def ai_move(board,depth, max_state):
    v_list = np.empty([1,0])
    i = 0
    for moves in board.legal_moves:
        c_board = board.copy()
        c_board.push(moves)
        #already 1-st level
        v = minimax_lookup(c_board,depth,float('-inf'),float('inf'),not max_state) #careful with depth
        i+=1
        #print(i,moves)
        v_list = np.concatenate((v_list,np.array([[v]])), axis = 1)
        #print(v_list)
    return v_list


turn_dict = {'b':False , 'w':True}

#prob of win+draw
def game_start():
    while not board.is_game_over():
        print(board)
        if turn_dict[player_side] == board.turn :
            print("Player's Turn")
            move = chess.Move.from_uci(input("Choose your move: "))
            #check move legality
            while move not in board.legal_moves:
                print("Illegal move!")
                move =  chess.Move.from_uci(input("Choose your move: "))
            board.push(move)
        else:
            print("Computer's Turn")
            print('Thinking...')
            if not turn_dict[player_side]: #ai is white
                #d = drichlet_count(data_np,board).reshape((1,list(board.legal_moves).__len__()))
                v = ai_move(board,2,True)
                print(v)
                move = list(board.legal_moves)[np.argmax(v)]
            else:# ai is black
                #d = drichlet_count(data_np,board).reshape((1,list(board.legal_moves).__len__()))
                v = ai_move(board,2,False)
                print(v)
                move = list(board.legal_moves)[np.argmin(v)]
            board.push(move)
            print(move)
        if board.is_checkmate():
            print("Checkmate")
        else:
            if board.is_check():
                print("Check")
            if board.is_stalemate():
                print("Stalemate")
    else:
        print (board.result())



def self_play():
    while not board.is_game_over():
        print(board)
        if turn_dict[player_side] == board.turn :
            print("Computer_1's Turn")
            print('Thinking...')
            move = list(board.legal_moves)[np.argmax(ai_move(board,0,False))]
            board.push(move)
            print(move)
        else:
            print("Computer_2's Turn")
            print('Thinking...')
            # ai is black
            move = list(board.legal_moves)[np.argmin(ai_move(board,0,True))]
            board.push(move)
            print(move)
        if board.is_checkmate():
            print("Checkmate")
        else:
            if board.is_check():
                print("Check")
            if board.is_stalemate():
                print("Stalemate")
    else:
        print (board.result())



sess.close()
