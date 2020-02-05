import numpy as np
import pandas as pd


def file_combine(start,finish):
    combine_df = pd.DataFrame()
    for i in range(start,finish,500):
        temp = pd.read_csv("~/Desktop/Chess/data/train_data{}".format(i), index_col = 0)
        combine_df = combine_df.append(temp, ignore_index=True)
        print (i)
    return combine_df

train_data_2013 = file_combine(499,41500)

temp = pd.read_csv("~/Desktop/Chess/data/train_data41737", index_col = 0)
train_data_2013 = train_data_2013.append(temp, ignore_index = True)




def file_combine(start,finish):
    combine_df = pd.DataFrame()
    for i in range(start,finish,10000):
        temp = pd.read_csv("~/Chess/piece_pos_data{}".format(i), index_col = 0)
        combine_df = combine_df.append(temp, ignore_index=True)
        print (i)
        #if (i+1) % 100000 == 0:
            #combine_df.to_csv("~/Desktop/Chess/data/piece_pos_checkpt")
            #print("check point done")
    return combine_df

piece_pos_data_2013 = file_combine(9999,3399999)

temp = pd.read_csv("~/Chess/data/piece_pos_data3406525", index_col = 0)
piece_pos_data_2013 = piece_pos_data_2013.append(temp, ignore_index = True)


def file_combine():
    combine_df = pd.DataFrame()
    for i in range(1,10):
        temp = pd.read_csv("./Chess/atk_map_{}".format(i), index_col = 0)
        combine_df = combine_df.append(temp, ignore_index=True)
        print (i)
        #if (i+1) % 100000 == 0:
            #combine_df.to_csv("~/Desktop/Chess/data/piece_pos_checkpt")
            #print("check point done")
    return combine_df

def file_combine():
    combine_np = np.empty([0,768])
    for i in range(1,10):
        temp = np.load("./Chess/atk_map_{}".format(i))
        combine_df = np.concatenate((combine_np,temp), axis = 0)
        print (i)
        #if (i+1) % 100000 == 0:
            #combine_df.to_csv("~/Desktop/Chess/data/piece_pos_checkpt")
            #print("check point done")
    return combine_df
