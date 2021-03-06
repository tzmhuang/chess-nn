from Desktop.chess-nn.minimax_lookup import *
from Desktop.chess-nn.evl_conv_3 import model_fn




def predict_input_fn(data):
    data = data.astype('float32')
    tmp = {'x':data}
    return tf.estimator.inputs.numpy_input_fn(x = tmp,num_epochs = 1, shuffle = False )


with tf.device('/gpu:0'):
    evl_conv_temp = tf.estimator.Estimator(
        model_fn = model_fn, model_dir = './DNN/evl_conv_5')



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
    Y = evl_conv_temp.predict(input_fn = predict_input_fn(in_data))
    return Y

def get_val(board):
    in_data = extract(board)
    graph = tf.get_default_graph()
    Y = graph.get_tensor_by_name("prob/Softmax:0")
    epsilon = tf.constant(0.00000000001)
    v = sess.run(Y, feed_dict={x:in_data})
    return v


# Number of moves from board_a to board_b
def nxtmv_count(data_np,board_a, board_b):
    ind = np.where((data_np == board_a).sum(axis = 1) == 64)[0]
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
                d = drichlet_count(data_np,board).reshape((1,list(board.legal_moves).__len__()))
                v = d+ai_move(board,1,True)
                move = list(board.legal_moves)[np.argmax(v)]
            else:# ai is black
                d = drichlet_count(data_np,board).reshape((1,list(board.legal_moves).__len__()))
                v = -d+ai_move(board,1,False)
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
