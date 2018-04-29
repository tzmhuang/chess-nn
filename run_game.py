from Desktop.chess-nn.minimax_lookup import *

meta_dir = "./chess_nn_result/evl_NN_Adam/model/evl_NN_Adam-23129.meta"
#meta_dir = "./DNN/evl_NN_Adam/model/evl_NN_Adam-23129.meta"


tf.reset_default_graph()

imported_meta = tf.train.import_meta_graph(meta_dir)
'''Initiating Variables...'''
sess = tf.Session()
imported_meta.restore(sess, tf.train.latest_checkpoint('./chess_nn_result/evl_NN_Adam/model/'))
# imported_meta.restore(sess, tf.train.latest_checkpoint('./DNN/evl_NN_Adam/model/'))
graph = tf.get_default_graph()
# W_0 = graph.get_tensor_by_name("layer_0/weights_0/W_0/read:0")
# W_1 = tf.get_tensor_by_name("layer_1/weights_1:0")
# W_2 = tf.get_tensor_by_name("layer_2/weights_2:0")
# W_3 = tf.get_tensor_by_name("layer_3/weights_3:0")
# b_0 = tf.get_tensor_by_name("layer_0/bias_0:0")
# b_1 = tf.get_tensor_by_name("layer_1/bias_1:0")
# b_2 = tf.get_tensor_by_name("layer_2/bias_2:0")
# b_3 = tf.get_tensor_by_name("layer_3/bias_3:0")


# graph = tf.get_default_graph()

'''Initiation Complete'''

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
                    print("max prune")
                    return v
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
                    return v
    return v

#need to consider lose or winning senarios
def evaluate(in_data):
    graph = tf.get_default_graph()
    Y = graph.get_tensor_by_name("prob/Softmax:0")
    epsilon = tf.constant(0.00000000001)
    v = sess.run(Y, feed_dict={x:in_data})
    return v[0,0]+v[0,2]
#prob of win+draw


def ai_move(board,depth, max_state):
    v_list = np.empty([1,0])
    i = 0
    for moves in board.legal_moves:
        c_board = board.copy()
        c_board.push(moves)
        #already 1-st level
        v = minimax_lookup(c_board,depth,float('-inf'),float('inf'), max_state) #careful with depth
        i+=1
        #print(i,moves)
        v_list = np.concatenate((v_list,np.array([[v]])), axis = 1)
        #print(v_list)
    return v_list


turn_dict = {'b':False , 'w':True}

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
                move = list(board.legal_moves)[np.argmax(ai_move(board,1,False))]
            else:# ai is black
                move = list(board.legal_moves)[np.argmin(ai_move(board,1,True))]
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
