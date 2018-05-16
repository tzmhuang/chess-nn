
import chess
import tensorflow as tf
import numpy as np
import h5py
from collections import OrderedDict


#board.legal_moves

# keys(): ['atk_map', 'board_set','castling','game_move_num',
#         'piece_pos', 'result','turn_move','flag']
piece_val = {'P':1, 'R':5, 'N':3, 'B':4, 'Q':9, 'K':100,
             'p':-1, 'r':-5, 'n':-3, 'b':-4, 'q':-9, 'k':-100,
             '.':0}

piece_val_2 = {'P':1, 'R':5, 'N':3, 'B':4, 'Q':9, 'K':100,
             'p':-1, 'r':-5, 'n':-3, 'b':-4, 'q':-9, 'k':-100}

piece_val_2 = OrderedDict(piece_val_2)

board_val_mp = {0:56, 1:57, 2:58, 3:59, 4:60, 5:61, 6:62, 7:63,
             8:48, 9:49, 10:50, 11:51, 12:52, 13:53, 14:54, 15:55,
             16:40, 17:41, 18:42, 19:43, 20:44, 21:45, 22:46, 23:47,
             24:32, 25:33, 26:34, 27:35, 28:36, 29:37, 30:38, 31:39,
             32:24, 33:25, 34:26, 35:27,36:28, 37:29, 38:30, 39:31,
             40:16, 41:17, 42:18, 43:19, 44:20, 45:21, 46:22, 47:23,
             48:8, 49:9, 50:10, 51:11, 52:12, 53:13, 54:14, 55:15,
             56:0, 57:1 ,58:2, 59:3, 60:4, 61:5, 62:6, 63:7}

#board_set (done)
def board_cvrt(board):
    Data_row = np.empty([1,0]) # initializing output of
    board_str = str(board)
    for piece in board_str:
        if piece == ' ' or piece == '\n':
            continue
        else:
            temp = np.array([[piece_val[piece]]])
            Data_row = np.concatenate((Data_row,temp), axis = 1)
    return Data_row

def castling_check_ks_w( board_state ):
    if bool(board_state.castling_rights & chess.BB_H1):
        return True
    else:
        return False

def castling_check_qs_w( board_state ):
    if bool(board_state.castling_rights & chess.BB_A1):
        return True
    else:
        return False

def castling_check_ks_b( board_state ):
    if bool(board_state.castling_rights & chess.BB_H8):
        return True
    else:
        return False

def castling_check_qs_b( board_state ):
    if bool(board_state.castling_rights & chess.BB_A8):
        return True
    else:
        return False

#castling (done)
def castling(board):
    ks_w = castling_check_ks_w( board )
    qs_w = castling_check_qs_w( board )
    ks_b = castling_check_ks_b( board )
    qs_b = castling_check_qs_b( board )
    Data_row = np.array([[int(ks_w),int(qs_w),int(ks_b),int(qs_b)]])
    return Data_row

#piece_pos (done)
def piece_pos(board_state):
    data = np.empty([1,0])
    for piece in piece_val_2:
        piece_num = piece_val_2[piece]
        row = (board_state == piece_num)
        data = np.concatenate((data,row), axis  = 1)
    return data

#turn (done)
def turn(board):
    return np.array([[int(board.turn)]])

#atk_map (done)
def atk_map(board_state,board):
    output = np.empty([1,0])
    for piece in piece_val_2:
        pos = np.where(board_state == piece_val_2[piece])[1]
        #piece_n = pos.size
        tmp_sqr = chess.SquareSet()
        for i in pos:
            board_pos = board_val_mp[i]
            tmp_sqr = tmp_sqr.union(board.attacks(board_pos))
        for j in range(64):
            temp = np.array([[int(board_val_mp[j] in tmp_sqr)]])
            output = np.concatenate((output, temp), axis = 1)
    return output


def game_phase(board):
    move_num = board.halfmove_clock
    if move_num <= 20:
        result = np.array([[1,0,0]])
    elif move_num > 60:
        result = np.array([[0,0,1]])
    else:
        result = np.array([[0,1,0]])
    return result

def conv_shape(data):
    tmp1 = data[:,0:64].transpose()
    tmp2 = data[:,64:128].transpose()
    tmp3 = data[:,128:192].transpose()
    tmp4 = data[:,192:256].transpose()
    tmp5 = data[:,256:320].transpose()
    tmp6 = data[:,320:384].transpose()
    tmp7 = data[:,384:448].transpose()
    tmp8 = data[:,448:512].transpose()
    tmp9 = data[:,512:576].transpose()
    tmp10 = data[:,576:640].transpose()
    tmp11 = data[:,640:704].transpose()
    tmp12 = data[:,704:768].transpose()
    return np.concatenate((tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9,tmp10,tmp11,tmp12), axis = 1)

#done
def extract(board):
    #g = game_phase(board)
    t = turn(board)
    t_conv = np.repeat(t,64).reshape(-1,1)
    c = castling(board)
    c_conv = np.repeat(c,64).reshape(4,-1).transpose()
    b = board_cvrt(board)
    b_conv = b.reshape(-1,1)
    p = piece_pos(b)
    p_conv = conv_shape(p)
    a = atk_map(b,board)
    a_conv = conv_shape(a)
    #result = np.concatenate((t,c,b,p,a), axis = 1)
    result = np.concatenate((b_conv,p_conv,a_conv,t_conv,c_conv), axis = 1)
    result = result.reshape(1,8,8,30)
    return result

#determine if the state is leaf
#done
def leaf(board):
    return not board.legal_moves  #True if legal move is empty
