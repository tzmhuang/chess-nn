from Desktop.chess-nn.minimax_lookup import *
from Desktop.chess-nn.evl_conv_3 import model_fn
import time
#meta_dir = "./chess_nn_result/evl_dense_final/model.ckpt-59900.meta"
#meta_dir = "./DNN/evl_NN_Adam/model/evl_NN_Adam-23129.meta"
meta_dir = "./DNN/evl_dense_final0/model.ckpt-106480.meta"
#meta_dir = "./evl_dense_final0/model.ckpt-106480.meta"

tf.reset_default_graph()

imported_meta = tf.train.import_meta_graph(meta_dir)


'''Initiating Variables...'''
sess = tf.Session()
imported_meta.restore(sess, tf.train.latest_checkpoint('./DNN/evl_dense_final0/'))
#imported_meta.restore(sess, tf.train.latest_checkpoint('./evl_dense_final0/'))
graph = tf.get_default_graph()
#W_0 = graph.get_tensor_by_name("layer_0/weights_0/W_0:0")
W_1 = graph.get_tensor_by_name("layer_1/weights_1/W_1:0")
W_2 = graph.get_tensor_by_name("layer_2/weights_2/W_2:0")
W_4 = graph.get_tensor_by_name("logits/weights_4/W_4:0")

#W_3 = tf.get_tensor_by_name("layer_3/weights_3:0")
#b_0 = graph.get_tensor_by_name("layer_0/bias_0/b_0:0")
b_1 = graph.get_tensor_by_name("layer_1/bias_1/b_1:0")
b_2 = graph.get_tensor_by_name("layer_2/bias_2/b_2:0")
b_4 = graph.get_tensor_by_name("logits/bias_4/b_4:0")
# b_3 = tf.get_tensor_by_name("layer_3/bias_3:0")
#bn0_mean = graph.get_tensor_by_name('BN0/moving_mean:0')
bn1_mean = graph.get_tensor_by_name('BN1/moving_mean:0')
bn2_mean = graph.get_tensor_by_name('BN2/moving_mean:0')

#bn0_var = graph.get_tensor_by_name('BN0/moving_variance:0')
bn1_var = graph.get_tensor_by_name('BN1/moving_variance:0')
bn2_var = graph.get_tensor_by_name('BN2/moving_variance:0')
#bn0_gamma = graph.get_tensor_by_name('BN0/gamma:0')
bn1_gamma = graph.get_tensor_by_name('BN1/gamma:0')
bn2_gamma = graph.get_tensor_by_name('BN2/gamma:0')
#bn0_beta = graph.get_tensor_by_name('BN0/beta:0')
bn1_beta = graph.get_tensor_by_name('BN1/beta:0')
bn2_beta = graph.get_tensor_by_name('BN2/beta:0')

x = tf.placeholder(tf.float32,name = 'input', shape = [None,1555] )
layer_1 = tf.matmul(x,W_1)+b_1
bn1 = tf.nn.batch_normalization(layer_1, mean =bn1_mean ,variance = bn1_var,
        offset =bn1_beta, scale =bn1_gamma, variance_epsilon = 10e-7 )
relu_1 = tf.nn.relu(bn1)
layer_2 = tf.matmul(relu_1,W_2)+b_2
bn2 = tf.nn.batch_normalization(layer_2, mean =bn2_mean ,variance = bn2_var,
        offset =bn2_beta, scale =bn2_gamma, variance_epsilon = 10e-7 )
relu_2 = tf.nn.relu(bn2)
logit_layer = tf.matmul(relu_2,W_4)+b_4
prediction = tf.nn.softmax(logit_layer)


player_side = input("Please choose your side(b/w): ")

difficulty = input("Choose Difficulty(1-10): ")
board = chess.Board()
#done? implement a-b pruning for memory saving
def minimax_lookup(board, depth, alpha, beta, max_state):
    if leaf(board) or depth == 0:
        in_board = extract(board)
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
                    #print("min prune")
                    return alpha
    return v

#need to consider lose or winning senarios
def evaluate(in_data):
    #start_time = time.time()
    Y = sess.run(prediction,feed_dict={x:in_data})
    #print (time.time()-start_time)
    return Y[0][0]+Y[0][2] #winning
    #return time.time-start_time

def get_val(board):
    in_data = extract(board)
    graph = tf.get_default_graph()
    Y = graph.get_tensor_by_name("prob/Softmax:0")
    epsilon = tf.constant(0.00000000001)
    v = sess.run(Y, feed_dict={x:in_data})
    return v


# Number of moves from board_a to board_b
# data_np = test_h['board_set'][:] and side of move
train_h = h5py.File("./DNN/train_data.h5")
test_h = h5py.File("./DNN/test_data.h5")
tr = np.concatenate((train_h['turn_move'][:],train_h['board_set'][:]),axis = 1)
tst = np.concatenate((test_h['turn_move'][:],test_h['board_set'][:]),axis = 1)
data_np = np.concatenate((tr,tst), axis = 0)


def nxtmv_count(data_np,board_a, board_b):
    ind = np.where( (( (data_np == board_a).sum(axis = 1)) == 65)[0])[0]
    if ind:
        ind = ind+1
        temp = data_np[ind]
        count = ((temp == board_b).sum(axis = 1)== 65).sum()
    else:
        count = 0
    return count

def next_move(board):
    result = np.empty([0,65], dtype=int)
    for move in board.legal_moves:
        tmp_board = board.copy()
        tmp_board.push(move)
        t = np.array([[int(tmp_board.turn)]])
        tmp = np.concatenate((t,board_cvrt(tmp_board)),axis = 1)
        result = np.concatenate((result,tmp), axis = 0)
    return result

def drichlet_count(data_np,board):
    count = np.empty([0,1])
    t = np.array([[int(board.turn)]])
    board_a = np.concatenate((t,board_cvrt(tmp_board)),axis = 1)
    legal_move = next_move(board)
    for board_b in legal_move:
        board_b = np.concatenate((t,board_cvrt(tmp_board)),axis = 1)
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



def sunfis_play():





sess.close()
