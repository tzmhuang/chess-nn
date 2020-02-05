source Chess/bin/activate
python3
import chess
import numpy as np
import chess.pgn
import pandas as pd
import h5py
from io import StringIO
from collections import OrderedDict



board_val_mp = {0:56, 1:57, 2:58, 3:59, 4:60, 5:61, 6:62, 7:63,
             8:48, 9:49, 10:50, 11:51, 12:52, 13:53, 14:54, 15:55,
             16:40, 17:41, 18:42, 19:43, 20:44, 21:45, 22:46, 23:47,
             24:32, 25:33, 26:34, 27:35, 28:36, 29:37, 30:38, 31:39,
             32:24, 33:25, 34:26, 35:27,36:28, 37:29, 38:30, 39:31,
             40:16, 41:17, 42:18, 43:19, 44:20, 45:21, 46:22, 47:23,
             48:8, 49:9, 50:10, 51:11, 52:12, 53:13, 54:14, 55:15,
             56:0, 57:1 ,58:2, 59:3, 60:4, 61:5, 62:6, 63:7}


piece_val_2 = OrderedDict((('P',1), ('R',5), ('N',3), ('B',4), ('Q',9), ('K',100),
             ('p',-1), ('r',-5), ('n',-3), ('b',-4), ('q',-9), ('k',-100)))



def attack_map(board,train_data,g_num, m_num):
    #one map for each piece  --> (6+6)*64
    alt_board = train_data[train_data.game_num == g_num].iloc[m_num-1,:]
    output = pd.Series()
    for piece in piece_val_2:
        alt_board_2 = alt_board.iloc[2:66]
        pos = alt_board_2[alt_board_2 == piece_val_2[piece]].index
        #piece_n = pos.size
        tmp_sqr = chess.SquareSet()
        for i in pos:
            board_pos = board_val_mp[int(i)]
            tmp_sqr = tmp_sqr.union(board.attacks(board_pos))
        for j in range(64):
            temp = pd.Series(int(board_val_mp[j] in tmp_sqr))
            output = output.append(temp, ignore_index=True)
    return output



def game_attack_map(game,game_num):
    board = chess.Board()
    Data_game = pd.DataFrame(columns = range(64*12), dtype = 'int8')
    move_num = 0
    for moves in game.main_line():
        board.push(moves)
        move_num += 1
        temp = attack_map(board,train_data,game_num, move_num)
        Data_game = Data_game.append(temp, ignore_index = True)
    return Data_game

def attack_map_df(pgn_df):
    #temp = pd.DataFrame()
    output = pd.DataFrame()
    for i in range(0,pgn_df.shape[0]): ###### error, redo data from 31500 to 31999, should be range(31500, 32000)
        index = i+41737
        game_num = index+1
        game = chess.pgn.read_game(StringIO(pgn_df.iloc[i,0]))
        cvrtd_game = game_attack_map(game,game_num)
        #temp = temp.append(cvrtd_game, ignore_index=True)
        output = output.append(cvrtd_game, ignore_index=True)
        #if (i+1) %500 == 0:
            #temp.to_csv("~/Chess/atk_map{}".format(i))
            #temp = pd.DataFrame()
            #print (i)
    #temp.to_csv("~/Chess/atk_map{}".format(i))
    return output
