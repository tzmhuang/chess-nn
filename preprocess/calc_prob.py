import numpy as np
import pandas as pd
import random

def calc_prob(pd_data, castling_data,):
    board_pos = pd_data.iloc[:,0:64]
    pd_data_x = pd.concat([board_pos, castling_data], axis = 1, ignore_index=True)
    pd_data_y = pd_data.iloc[:,64:65]
    #completed_states = pd.DataFrame(np.zeros(shape = (1,pd_data_x.shape[1])))
    prob_col = pd.DataFrame(np.zeros(shape = (pd_data_x.shape[0],3)))
    for i in range(pd_data_x.shape[0]):
        if (prob_col.iloc[i].sum()) == 0 :
            temp_pd = (pd_data_x == pd_data_x.iloc[i]).sum(axis = 1)
            y_index = tedatamp_pd[temp_pd == 68].index  #caution: temp_pd == ?
            #index with same state
            same_state_y = pd_data_y.iloc[y_index,:]
            prob_w = (((same_state_y ==  1).sum())/same_state_y.size).reset_index(drop = True)[0]
            prob_l = (((same_state_y == -1).sum())/same_state_y.size).reset_index(drop = True)[0]
            prob_d = (((same_state_y == 0).sum())/same_state_y.size).reset_index(drop = True)[0]
            prob_col.iloc[y_index] = (prob_w, prob_l, prob_d)
            #print("0")
        else:
            continue
        if (i+1)%500000 == 0:
            prob_col.to_csv("~/Chess/prob_col{}".format(i))
            print(i)
        if (i == 3406525):#change as needed
            prob_col.to_csv("~/Chess/prob_col_2013")
    return prob_col

pd_data = pd.read_csv("~/Chess/train_data_2013")
castling_data =  pd.read_csv("~/Chess/castling_data_2013")

'''
Using non-duplicate list of states
Parallel Computing: achievable
'''

data = pd.read_csv("Desktop/Chess/data/chess_data_2013", index_col = 0)

def calc_prob(data, nodup_list, start_index, finish_index):
    pd_data_x = data.iloc[:,0:67]
    pd_data_y = data.iloc[:,67:68]
    #completed_states = pd.DataFrame(np.zeros(shape = (1,pd_data_x.shape[1])))
    prob_col = pd.DataFrame(np.zeros(shape = (nodup_list.shape[0],3)))
    for i in range(start_index, finish_index):
            temp_pd = (pd_data_x == nodup_list.iloc[i,0:68]).sum(axis = 1)
            same_state_y = pd_data_y[temp_pd == 68]
            prob_w = (((same_state_y ==  1).sum())/same_state_y.size).reset_index(drop = True)[0]
            prob_l = (((same_state_y == -1).sum())/same_state_y.size).reset_index(drop = True)[0]
            prob_d = (((same_state_y == 0).sum())/same_state_y.size).reset_index(drop = True)[0]
            prob_col.iloc[i] = (prob_w, prob_l, prob_d)
            #print("0")
            if (i+1)%1 == 0:
                #prob_col.to_csv("~/Chess/prob_col{}".format(i))
                print(i)
            # if (i == 3406525):#change as needed
            #     prob_col.to_csv("~/Chess/prob_col_2013")
    return prob_col

# pd_data = data
# castling_data = castling
# noduptoy = data_x_n_castling_nodup
# start = 0
# finish = 100000
#
# calc_prob(data, castling, data_x_n_castling_nodup, 0, 100000)
#
# if __name__ == __main__:
#     calc_prob(pd_data, castling_data)

#5556 hours no parralell

'''
calc_prob algorithm requires running time over 5555 hours if no Parallel
not feasible
hence use sampling approach to estimate probability
'''

board_pos = data.iloc[:,0:64]
pd_data_x = pd.concat([board_pos, castling], axis = 1, ignore_index=True)
pd_data_y = data.iloc[:,64:65]
_data_ = pd.concat([pd_data_x,pd_data_y], axis = 1,  ignore_index=True)

def sampling(data,sample_size):
    rand_index = random.sample(range(2896958),sample_size)
    return data.iloc[rand_index,:]


'''
testing running time
_data_ = pd.read_csv("Desktop/Chess/data/chess_data_2013", index_col = 0)
'''

#sample size 10000
#time/1_completion = 1/50 s
data10000 = _data_.iloc[0:10000,:]
calc_prob(data10000, data_x_n_castling_nodup, 0, 100000)

#sample size 50000
#time/1_completion = 1/10 s
data50000 = _data_.iloc[0:50000,:]
calc_prob(data50000, data_x_n_castling_nodup, 0, 100000)

#sample size 100000
#time/1_completion = 1/5 s
data100000 = _data_.iloc[0:100000,:]
calc_prob(data100000, data_x_n_castling_nodup, 0, 100000)

#sample size 500000
#time/1_completion = 1 s
data500000 = _data_.iloc[0:500000,:]
calc_prob(data500000, data_x_n_castling_nodup, 0, 100000)
