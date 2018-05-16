import tensorflow as tf
import chess
import numpy as np
import pandas as pd
import time

# sts_dir = "./Desktop/STS[1-13]/STS11.epd"
# epd_data = pd.read_csv(sts_dir, sep= '\r', header = None)
#
# board = chess.Board()
# dtype = [('move','S10'),('value',float)]
#
# epd_board = board.from_epd(epd_data.iloc[0,0])

def bm_rank(depth, epd_board,max_state):
    #start = time.time()
    v = ai_move(epd_board[0],depth, max_state).reshape(-1,1)
    move = np.array(list(epd_board[0].legal_moves),dtype = 'str').reshape(-1,1)
    temp_df = np.concatenate((move,v),1)
    temp_df = pd.DataFrame(temp_df)
    df_s = temp_df.sort_values(by=[1], ascending = False)
    df_s = df_s.reset_index(drop = True)
    bm = str(epd_board[1]['bm'][0])
    rank = int(df_s[df_s[0] == bm].index.values)
    #print(time.time() - start)
    return rank

#loop_through files ranks = ranks+1

def depth_eval(depth):
    depth_rank = np.empty([10,0])
    for num in range(1,14):
        start = time.time()
        #sts_dir = "./Desktop/STS[1-13]/STS{}.epd".format(num)
        sts_dir = "./DNN/sts/STS{}.epd".format(num)
        epd_data = pd.read_csv(sts_dir, sep= '\r', header = None)
        file_ranks = list()
        for i in range(10):
            epd_board = board.from_epd(epd_data.iloc[i,0][:epd_data.iloc[i,0].find(';')])
            rank = bm_rank(depth, epd_board,epd_board[0].turn)
            file_ranks.append(rank)
        temp = np.array(file_ranks).reshape(-1,1)
        #print(temp)
        depth_rank = np.concatenate((depth_rank,temp), axis = 1)
        print('-----',time.time() - start)
    result = pd.DataFrame(depth_rank)
    # result.to_csv('./chess_nn_result/STS_eval/sts{}'.format(depth))
    result.to_csv('./DNN/STS_eval/sts{}'.format(depth))
    return depth_rank

board = chess.board()
for i in range(0,3):
    start = time.time()
    depth_eval(i)
    print('-',time.time() - start)
