import chess
import numpy as np
import chess.pgn
import pandas as pd
from io import StringIO                                                         #for PGN string parsing

#pgn = open("Desktop/Chess/ficsgamesdb_201701_standard_nomovetimes_1511264.pgn") #Total number of games: 104035
./google-cloud-sdk/bin/gcloud compute ssh --zone=asia-east1-a nn-instance1

# from bucket to VM
gsutil cp gs://[BUCKET_NAME]/[OBJECT_NAME] [OBJECT_DESTINATION]
gsutil cp gs://chess-nn/pgn_data_titled_2013 ~/Chess
gsutil cp gs://chess-nn/data ~/Chess
gsutil cp gs://chess-nn/data/train_data_2013 ~/Chess
gsutil cp gs://chess-nn/data/atk_map_* /home/huangtom2/Chess

gsutil cp gs://chess-nn/libcudnn7-dev_7.1.3.16-1+cuda8.0_amd64.deb	 ~/
gsutil cp gs://chess-nn/libcudnn7_7.1.3.16-1+cuda8.0_amd64.deb	 ~/

gsutil cp gs://chess-nn/cuDNN/libcudnn7_7.1.3.16-1+cuda9.1_amd64.deb	 ~/
gsutil cp gs://chess-nn/cuDNN/libcudnn7-dev_7.1.3.16-1+cuda9.1_amd64.deb	 ~/
gsutil cp gs://chess-nn/cuDNN/libcudnn7-doc_7.1.3.16-1+cuda9.1_amd64.deb ~/




##set up CUDA-nn
sudo dpkg -i libcudnn7-dev_7.1.3.16-1+cuda8.0_amd64.deb
sudo dpkg -i libcudnn7_7.1.3.16-1+cuda8.0_amd64.deb
sudo apt-get install cuda-command-line-tools

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

#from VM to bucket?
gsutil cp [LOCAL_OBJECT_LOCATION] gs://[DESTINATION_BUCKET_NAME]/
gsutil cp  ~/Chess/499 gs://chess-nn/
gsutil cp  ~/DNN gs://chess-nn/


# from VM to local
gcloud compute copy-files [INSTANCE_NAME]:[REMOTE_FILE_PATH] [LOCAL_FILE_PATH]
gcloud compute copy-files nn-instance1:~/Chess/train_data_2013 ~/Desktop

gcloud compute copy-files nn-instance1:~/Chess/castling_* ~/Desktop/Chess/data
gcloud compute copy-files nn-instance1:/home/huangtom2/Chess/train_data{32499..41499..500} ~/Desktop/Chess/data
gcloud compute copy-files nn-instance1:/home/huangtom2/Chess/game_move_num_* ~/Desktop/Chess/data
gcloud compute copy-files nn-instance1:/home/huangtom2/Chess/at* ~/Desktop/Chess/data
gcloud compute copy-files nn-instance1:/home/huangtom2/chess_data.gz ~/Desktop/Chess/data
gcloud compute copy-files nn-instance1:/home/huangtom2/Chess/flag.npy ~/Desktop
gcloud compute copy-files nn-instance1:/home/huangtom2/DNN/evl_conv_1 ./



#from local to VM
gcloud compute scp [LOCAL_FILE_PATH] [INSTANCE_NAME]:~/
gcloud compute copy-files ~/Desktop/Chess/data/piece_pos_data_check_1 nn-instance1:/home/huangtom2/Chess --zone asia-east1-a
gcloud compute copy-files ~/Desktop/Chess/data/game_move_num root@nn-instance1:/home/huangtom2/Chess --zone asia-east1-a

gcloud compute scp ~/Desktop/Chess/data/game_move_num nn-instance1:/home/huangtom2/Chess
# check maintanance
curl http://metadata.google.internal/computeMetadata/v1/instance/maintenance-event -H "Metadata-Flavor: Google"
'''
Meta data:
    1. 41738 total games played
    2. 3406526 total game states
    3. 3406526 total castling states
'''
# pgn = open("Desktop/Chess/tmp.pgn")
# line_br_detector = 0
#
# for line in pgn:
#     if line == '\n':
#         line_br_detector += 1
#

def pgn_file_reader(file_dir):
    pgn = open(file_dir)
    line_br_detector = 0
    pgn_data_frame = pd.DataFrame()
    #temp0 = pd.Series()
    temp1 = pd.Series()
    for line in pgn.read().splitlines():                                        #Forming DataFrame with Pandas
        if line == "":
            line_br_detector += 1
        if line_br_detector%2 != 0:
            temp1 = pd.Series(line)
        if line_br_detector > 0 and line_br_detector%2 == 0:
            pgn_data_frame = pgn_data_frame.append(temp1, ignore_index=True)
            temp1 = pd.Series()
            line_br_detector = 0
            continue
        #temp0 = pd.Series(line)
        #temp1 = temp1.append(temp0, ignore_index=True)
    pgn.close()
    return pgn_data_frame


#altered, extract head information
def pgn_file_reader(file_dir):
    pgn = open(file_dir)
    line_br_detector = False
    pgn_data_frame = pd.DataFrame()
    #temp1 = pd.Series()
    temp0 = ''
    for line in pgn.read().splitlines():                                        #Forming DataFrame with Pandas
        if line == "":
            #print (0)
            line_br_detector  = not line_br_detector
        if line_br_detector:
            #print (1)
            pgn_data_frame = pgn_data_frame.append(temp0, ignore_index=True)
            temp0 = ''
            line_br_detector  = not line_br_detector
            #temp1 = pd.Series()
        else:
            #print (2)
            temp0 = temp0+'\n'+line
            temp0 = pd.Series(temp0)
            #temp1 = temp1.append(temp0,ignore_index = True)
            continue
        #temp0 = pd.Series(line)
        #temp1 = temp1.append(temp0, ignore_index=True)
    pgn.close()
    return pgn_data_frame




def



pgn_data = pgn_file_reader("Desktop/Chess/ficsgamesdb_201701_standard_nomovetimes_1511264.pgn")

# pgn_string = DataX[18]
# game_string = pgn_string[i]         #i in 0 to n-1
#
# _game_ = chess.read.game(StringIO(game_string))

def run_game(game):
    board = chess.Board()
    for moves in game.main_line():
        board.push(moves)
    return board

def check_counter(pgn_df):
    count = 0
    for i in range(0,pgn_df.size):
        tmp_game = chess.pgn.read_game(StringIO(pgn_df[0][i]))
        board = run_game(tmp_game)
        print (board.result())
        if board.result() != '*':
            count += 1
    return count


def elo_extract(header_np):
    result = np.empty([0,2])
    for i in range(header_np.size):
        pgn_string = header_np[i,0]
        pgn = StringIO(pgn_string)
        game = chess.pgn.read_game(pgn)
        black_elo = int(game.headers['BlackElo'])
        white_elo = int(game.headers['WhiteElo'])
        tmp = np.concatenate((np.array([[white_elo]]),np.array([[black_elo]])), axis = 1)
        result = np.concatenate((result,tmp), axis = 0)
    return result



elo2000 = (black_elo>=2000) *(white_elo>=2000)
np.where(train_game_num in ind2000)


piece_val = {'P':1, 'R':5, 'N':3, 'B':4, 'Q':9, 'K':100,
             'p':-1, 'r':-5, 'n':-3, 'b':-4, 'q':-9, 'k':-100,
             '.':0}


################################################################################
# board_cvrt    game_cvrt   move_df
################################################################################
data = pd.read_csv("../../huangtom/Chess/pgn_data_titled_2013")

# board_cvrt function
# input: board_state( type: chess.Board )
# output: a row of pd.Series,
# 0-63 as chess piece representations,
# 64 as castling_flag_w
# 65 being game result.
def board_cvrt( board_state, game ):
    Data_row = pd.Series() # initializing output of
    board_str = str(board_state)
    for piece in board_str:
        if piece == ' ' or piece == '\n':
            continue
        else:
            temp = pd.Series([piece_val[piece]])
            Data_row = Data_row.append(temp, ignore_index=True)
    if game.headers['Result'] == "1-0":
        result = pd.Series([1])
    elif game.headers['Result'] == "1/2-1/2":
        result = pd.Series([0])
    else :
        result = pd.Series([-1])
    Data_row = Data_row.append(result,ignore_index=True)
    return Data_row


#input: board state
#output: pd.Series (768) of binary position representation of each piece
def board_cvrt_sqr( board_state, piece ):
    Data_row = pd.Series() # initializing output of
    board_str = str(board_state)
    for i in board_str:
        if i == ' ' or i == '\n':
            continue
        elif i == '.':
            temp = pd.Series([piece_val[i]])
            Data_row = Data_row.append(temp, ignore_index=True)
        else:
            temp = pd.Series(i)
            Data_row = Data_row.append(temp, ignore_index=True)
    return Data_row


#game_cvrt() function: convert game into pandas data frame
#input: chess.pgn.game object
#output:  pandas.DataFrame, each row is out put of board_cvrt()
def game_cvrt(game):
    board = chess.Board()
    Data_game = pd.DataFrame()
    for moves in game.main_line():
        board.push(moves)
        temp = board_cvrt(board,game)
        Data_game = Data_game.append(temp, ignore_index = True)
    return Data_game.astype('int8')

def move_df(pgn_df):
    temp = pd.DataFrame()
    output = pd.DataFrame()
    for i in range(38000,pgn_df.shape[0]): ###### error, redo data from 31500 to 31999, should be range(31500, 32000)
        game = chess.pgn.read_game(StringIO(pgn_df.iloc[i,0]))
        cvrtd_game = game_cvrt(game)
        temp = temp.append(cvrtd_game, ignore_index=True)
        output = output.append(cvrtd_game, ignore_index=True)
        if (i+1) %500 == 0:
            temp.to_csv("~/Chess/train_data{}".format(i))
            temp = pd.DataFrame()
            print (i)
    temp.to_csv("~/Chess/train_data{}".format(i))
    return output

pgn_data = move_df(data)
pgn_data.to_csv("~/Chess/train_data_final")

################################################################################
# game_number
################################################################################

def move_counter(game, game_num):
    board = chess.Board()
    counter_game = pd.DataFrame(columns = ["game_num","move_num"],dtype = "int8")
    move_num = 0
    for moves in game.main_line():
        board.push(moves)
        move_num +=1
        temp = pd.Series({'game_num':game_num,'move_num':move_num})
        counter_game = counter_game.append(temp, ignore_index = True)
    return counter_game

def game_num_df(pgn_df):
    #temp = pd.DataFrame(columns = ["game_num","move_num"], dtype = "int8")
    output = pd.DataFrame(columns = ["game_num","move_num"],dtype = "int8")
    for i in range(pgn_df.shape[0]): ###### error, redo data from 31500 to 31999, should be range(31500, 32000)
        game = chess.pgn.read_game(StringIO(pgn_df.iloc[i,0]))
        index = i+41737
        counter_df = move_counter(game,index+1)
        #temp = temp.append(counter_df, ignore_index=True)
        output = output.append(counter_df, ignore_index=True)
        if (i+1) %1== 0:
            #temp.to_csv("~/Chess/train_data{}".format(i))
            #temp = pd.DataFrame()
            print (i)
    #temp.to_csv("~/Chess/train_data{}".format(i))
    return output

game_move_num = game_num_df(pgn_data)

################################################################################
# castling check
################################################################################
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

def castling_row( board_state):
    ks_w = castling_check_ks_w( board_state )
    qs_w = castling_check_qs_w( board_state )
    ks_b = castling_check_ks_b( board_state )
    qs_b = castling_check_qs_b( board_state )
    Data_row = pd.Series([int(ks_w),int(qs_w),int(ks_b),int(qs_b)])
    return Data_row

def castling_game_cvrt(game):
    board = chess.Board()
    Data_game = pd.DataFrame()
    for moves in game.main_line():
        board.push(moves)
        temp = castling_row(board)
        Data_game = Data_game.append(temp, ignore_index = True)
    return Data_game.astype('int8')

def castling_move_df(input_df):
    temp = pd.DataFrame()
    output = pd.DataFrame()
    for i in range(0,input_df.shape[0]):
        game = chess.pgn.read_game(StringIO(input_df.iloc[i,0]))
        cvrtd_game = castling_game_cvrt(game)
        temp = temp.append(cvrtd_game, ignore_index=True)
        output = output.append(cvrtd_game, ignore_index=True)
        if (i+1) %500 == 0:
            temp.to_csv("~/Chess/castling_col_{}".format(i))
            temp = pd.DataFrame()
            print (i)
    temp.to_csv("~/Chess/castling_col_{}".format(i))
    return output

castling_data = castling_move_df(data)

################################################################################
# moving side check white is 1
################################################################################

def turn_state(board_state):
    return pd.Series(int(board_state.turn))

def turn_game(game):
    board = chess.Board()
    Data_game = pd.DataFrame()
    for moves in game.main_line():
        board.push(moves)
        temp = turn_state(board)
        Data_game = Data_game.append(temp, ignore_index=True)
    return Data_game.astype('int8')

def turn_move_df(input_df):
    temp = pd.DataFrame()
    output = pd.DataFrame()
    for i in range(0,input_df.shape[0]):
        game = chess.pgn.read_game(StringIO(input_df.iloc[i,0]))
        cvrtd_game = turn_game(game)
        output = output.append(cvrtd_game, ignore_index=True)
        if (i+1) %500 == 0:
            print (i)
    return output

data = pd.read_csv("./Chess/pgn_data_titled_2013", index_col = 0)

data1 = data.iloc[0:5000]
data2 = data.iloc[5000:10000]
data3 = data.iloc[10000:15000]
data4 = data.iloc[15000:20000]
data5 = data.iloc[20000:25000]
data6 = data.iloc[25000:30000]
data7 = data.iloc[30000:35000]
data8 = data.iloc[35000:41738]

move_df8 = turn_move_df(data8)
move_df8.to_csv("./Chess/move_df_8")
################################################################################
# is_pinned
################################################################################

################################################################################
# attacks
################################################################################

################################################################################
# attackers
################################################################################
#pd_data_x = Data_df.iloc[:,0:65]
#pd_data_y = Data_df.iloc[:,65:66]
def calc_prob(pd_data):
    pd_data_x = pd_data.iloc[:,0:64]
    pd_data_y = pd_data.iloc[:,64:65]
    completed_states = pd.DataFrame(np.zeros(shape = (1,pd_data_x.shape[1])))
    prob_col = pd.DataFrame(np.zeros(shape = (pd_data_x.shape[0],3)))
    for i in range(pd_data_x.shape[0]):
        if (((completed_states == pd_data_x.iloc[i]).sum(axis = 1) == 64).sum()) == 0:  #caution: pd_data_x.iloc[i]).sum(axis = 1) == ?
            temp_pd = (pd_data_x == pd_data_x.iloc[i]).sum(axis = 1)
            y_index = temp_pd[temp_pd == 65].index  #caution: temp_pd == ?
             #index with same state
            same_state_y = pd_data_y.iloc[y_index,:]
            prob_w = (((same_state_y ==  1).sum())/same_state_y.size).reset_index(drop = True)[0]
            prob_l = (((same_state_y == -1).sum())/same_state_y.size).reset_index(drop = True)[0]
            prob_d = (((same_state_y == 0).sum())/same_state_y.size).reset_index(drop = True)[0]
            prob_col.iloc[y_index] = (prob_w, prob_l, prob_d)
            completed_states = completed_states.append(pd_data_x.iloc[i], ignore_index=True)
            print("a")
        else:
            continue
            #calculated the probability of winning losing and drawing
            #create dataframe with each state
            #record state already visited to avoid repetition
            #output result for each line of input
    #output = pd.concat([pd_data, prob_col], axis = 1)
    return prob_col

calc_prob(train_data499)
prob_col.to_csv("~/Chess/prob_col_499")
#done-list: 499,




#data_y[:,].reshape(16231,1)

# feature to extract:
# 1. number of each piece
# 2. positon of each piece
# 3. piece mobility

def side_to_move(board):
    return int(board.turn) #true is white


game.board().has_legal_en_passant()

def piece_position(board):
    piece_pos = pd.Series()
    for j in range(0,2):
        for i in range(1,7):
            square = board.pieces(i,j)
            piece = chess.Piece(i,j).symbol()
            piece_rep = board_cvrt_sqr(square, piece)
            piece_pos = piece_pos.append(piece_rep, ignore_index = True)
    return piece_pos.astype('int8')

#function equivilant to game_cvrt() but using board_cvrt_sqr() instead
def game_cvrt_sqr(game):
    board = chess.Board()
    Data_game = pd.DataFrame()
    for moves in game.main_line():
        board.push(moves)
        temp = piece_position(board)
        Data_game = Data_game.append(temp, ignore_index=True)
    return Data_game.astype('int8')

def piece_pos_df(pgn_df):
    temp = pd.DataFrame()
    for i in range(0,pgn_df.size):
        game = chess.pgn.read_game(StringIO(pgn_df.iloc[i,0]))
        temp = temp.append(game_cvrt_sqr(game), ignore_index=True)
    return temp



################################################################################
# FUNCTION: piece_pos_sep()
################################################################################
from collections import OrderedDict
piece_val_2 = {'P':1, 'R':5, 'N':3, 'B':4, 'Q':9, 'K':100,
             'p':-1, 'r':-5, 'n':-3, 'b':-4, 'q':-9, 'k':-100}
piece_val_2 = (('P',1),('a',2),('c',3),('t',4),('h',5))
temp = OrderedDict(piece_val_2)
for i in piece_val_2:
    print(i)
#piece_pos_sep()
#input: move_df_x, position only , shape should be (?,64), no castling term and flag!!
#output: pd.DataFrame with shape (12,64) for each row of input
## potential inprovement: do not ignore index for easy screening of specific piece later
def piece_pos_sep(move_df):
    #output = pd.DataFrame()
    check_pt = pd.DataFrame()
    for i in range(0,3):
        temp = pd.Series()
        for piece in piece_val_2:
            piece_num = piece_val_2[piece]
            #row = (move_df.iloc[i] == piece_num)
            row = (move_df.iloc[i] == piece_num)
            temp = temp.append(row, ignore_index=True)
        check_pt = check_pt.append(temp, ignore_index=True)
        #output = output.append(temp, ignore_index=True)
        if (i+1)%10000 == 0:
            check_pt = check_pt.astype('int8')
            check_pt.to_csv("~/Chess/piece_pos_data{}".format(i))
            print (i+2850000) #remember to change!!!!!!!!!
            check_pt = pd.DataFrame()
            continue
        check_pt = check_pt.astype('int8')
        check_pt.to_csv("~/Chess/piece_pos_data{}".format(i))
    return #output.astype('int8')


# numpy version
def piece_pos_sep(move_np_x):
    result = np.empty([0,768], dtype='int8')
    for i in range(0,move_np_x.shape[0]):
        temp = np.empty([1,0],dtype = "int8")
        for piece in piece_val_2:
            piece_num = piece_val_2[piece]
            row = (move_np_x[i:i+1] == piece_num)
            temp = np.concatenate((temp,row), axis = 1)
        result = np.concatenate((result,temp),axis = 0)
        if i%(i+1) == 10000:
            print (i)
    return result



'''
import h5py
import numpy as np
train_h = h5py.File("//Volumes/DiskA/train_data.h5")   #2384568
test_h = h5py.File("//Volumes/DiskA/test_data.h5") #1021958
# train_h = h5py.File("./DNN/train_data.h5")
# test_h = h5py.File("./DNN/test_data.h5")
8-cores:
1. r1 = piece_pos_sep(np1) np1 = train_h['board_set'][0:477000,:]
2. r2 = piece_pos_sep(np2) np2 = train_h['board_set'][477000:954000,:]
3. r3 = piece_pos_sep(np3) np3 = train_h['board_set'][954000:1431000,:]
4. r4 = piece_pos_sep(np4) np4 = train_h['board_set'][1431000:1908000,:]
5. r5 = piece_pos_sep(np5) np5 = train_h['board_set'][1908000:2384568,:]
6. r6 = piece_pos_sep(np6) np6 = test_h['board_set'][0:341000,:]
7. r7 = piece_pos_sep(np7) np7 = test_h['board_set'][341000:682000,:]
8. r8 = piece_pos_sep(np8) np8 = test_h['board_set'][682000:1021958,:]
'''



#1. piece_pos_sep(data.iloc[0:80000,0:64])          CPU4
#2. piece_pos_sep(data2.iloc[80000:160000,0:64])    CPU1
#3. piece_pos_sep(data3.iloc[160000:240000,0:64])
#done


#second phase
#1. piece_pos_sep(data.iloc[240000:1300000,0:64])    #to 1289999      CPU4
#2. piece_pos_sep(data2.iloc[1300000:1800000,0:64])   #done
#3. piece_pos_sep(data3.iloc[1800000:2300000,0:64])     #done

#third phase
#4. piece_pos_sep(data2.iloc[2300000:2850000,0:64])     #to 2369999
#6. piece_pos_sep(data3.iloc[2850000:3406526,0:64])




source Chess/bin/activate

python3

import numpy as np
import pandas as pd

data1 = pd.read_csv("Chess/train_data_2013", index_col = 0).iloc[240000:1300000,0:64]
data2 = pd.read_csv("Chess/train_data_2013", index_col = 0).iloc[2300000:2850000,0:64]
data3 = pd.read_csv("Chess/train_data_2013", index_col = 0).iloc[2850000:3406526,0:64]





piece_pos_sep(train_500_x.iloc[0:3,0:64])

data = pd.read_csv("Desktop/Chess/train_500", index_col = 0)
train_500_x = data.iloc[:,0:65]
train_500_y = data.iloc[:,65:66]


piece_pos_500 = piece_pos_sep(train_500_x.iloc[:,0:64])



################################################################################
# FUNCTION: Creating a unique list of all available states
# DONE
################################################################################

################################################################################
# Convert result into 3 column array (win,lose,draw)
#
################################################################################
source Chess/bin/activate
python3

import numpy as np
import h5py

def r2y(result_np):
    y = np.empty([0,3], dtype = "int8")
    for i in range(0,result_np.shape[0]):
        if result_np[i] == 1:
            temp = np.array([[1,0,0]])
        elif result_np[i] == -1:
            temp = np.array([[0,1,0]])
        elif result_np[i] == 0:
            temp = np.array([[0,0,1]])
        else:
            return -1
        y = np.concatenate((y,temp),axis = 0)
        if (i+1)%5000 == 0: print (i)
    return y

h = h5py.File("./Chess/chess_data.h5")
result = h['result']
result_np = np.array(result)

result_np1 = result_np[0:425000]
y1 = r2y(result_np1)
np.save("./Chess/y1", y1)

result_np2 = result_np[425000:900000]
y1 = r2y(result_np2)
np.save("./Chess/y2", y1)

result_np3 = result_np[900000:1350000]
y1 = r2y(result_np3)
np.save("./Chess/y3", y1)

result_np4 = result_np[1350000:1800000]
y1 = r2y(result_np4)
np.save("./Chess/y4", y1)

result_np5 = result_np[1800000:2250000]
y1 = r2y(result_np5)
np.save("./Chess/y5", y1)

result_np6 = result_np[2250000:2700000]
y1 = r2y(result_np6)
np.save("./Chess/y6", y1)

result_np7 = result_np[2700000:3150000]
y1 = r2y(result_np7)
np.save("./Chess/y7", y1)

result_np8 = result_np[3150000:3406526]
y1 = r2y(result_np8)
np.save("./Chess/y8", y1)

def file_combine(start,finish):
    combine_np = np.empty([0,3],dtype = "int8")
    for i in range(start,finish):
        temp = np.load("./Chess/y{}.npy".format(i))
        combine_np = np.concatenate((combine_np,temp), axis = 0)
        print (i)
        #if (i+1) % 100000 == 0:
            #combine_df.to_csv("~/Desktop/Chess/data/piece_pos_checkpt")
            #print("check point done")
    return combine_np


################################################################################
'''Number of White and Black piece on board'''
################################################################################

def wht_blk_piece_num(h5_ptr):
    data = h5_ptr.piece_pos[:,:]
    result = np.empty([0,2], type = 'int8')
    for i in shape.data[0]:
        w_num = np.array([[data[i][0:384].sum()]])
        b_num = np.array([[data[i][384:768].sum()]])
        temp = np.concatenate((w_num,b_num),axis = 1)
        result = np.concatenate((result,temp), axis = 0)
    return result



################################################################################
'''
Organize data as HDF5 data file using h5py package
'''
################################################################################
import numpy as np
import h5py
import pandas as pd

def pd2np_merg(start,fin):
    result = np.empty([0,1], dtype = 'int8')
    for i in range(start,fin):
        temp = pd.read_csv("./Chess/move_df_{}".format(i), index_col = 0, dtype = 'int8')
        temp = np.array(temp)
        result = np.concatenate((result,temp), axis = 0)
        print (i)
    return result


def pd2h5(np_data,ds_name):
    with h5py.File("./Chess/data") as h:
        tmp = h.create_dataset(ds_name,data = np)







2, 90113,91547,87779 == 269439 ok
26, 19526,21966, 21062, 19780, 4644 == 86978 ok
27, true:87836, get:87836 ok
28, true:86503, get:86502 ok
29, true:83107, get:83107 ok
30, true:88119, get:88119 ok
32, true:83514, get:83514 ok

total: 3406485

true - real
p33: 79108-79067 = 131 wrong
p32: 83514
def count(start,end):
    t = 0
    for i in range(start,end):
        t = t+train_data[train_data.game_num == i].shape[0]
    return t

def moves_n():
    for i in range(40741,41738):
        a = train_data[train_data.game_num == i].move_num
        print (max(a))

# select/filter from h5 dataset, and create new dataset
def h5_select(h5_ptr, obj, new_dir):
    h = h5py.File(new_dir)
    obj_data = h5_ptr[obj][:]
    logic = np.multiply(obj_data >20, obj_data <= 60)
    #logic = obj_data > 60
    ind = np.where(logic)[0]
    print(ind)
    move_num = h5_ptr['move_num'][:] #1
    move_num = move_num[ind]
    h['move_num'] = move_num
    print(move_num)
    game_phase = h5_ptr['game_phase'][:] #3
    game_phase = game_phase[ind]
    h['game_phase'] = game_phase
    turn_move = h5_ptr['turn_move'][:] #1
    turn_move = turn_move[ind]
    h['turn_move'] = turn_move
    castling = h5_ptr['castling'][:] #4
    castling = castling[ind]
    h['castling'] = castling
    board_set = h5_ptr['board_set'][:] #64
    board_set = board_set[ind]
    h['board_set'] = board_set
    piece_pos = h5_ptr['piece_pos'][:] #768
    piece_pos = piece_pos[ind]
    h['piece_pos'] = piece_pos
    atk_map = h5_ptr['atk_map'][:] #768
    atk_map = atk_map[ind]
    h['atk_map'] = atk_map
    flag = h5_ptr['flag'][:] #3
    flag = flag[ind]
    h['flag'] = flag
    h.close()
    return




################################################################################
'''
1.Giving every uniq pos an index
2.Group by pos --> count number of a-->b
'''
################################################################################
piece_val = {'P':1, 'R':5, 'N':3, 'B':4, 'Q':9, 'K':100,
             'p':-1, 'r':-5, 'n':-3, 'b':-4, 'q':-9, 'k':-100,
             '.':0}

def board_cvrt(board):
    Data_row = np.empty([1,0], dtype=int) # initializing output of
    board_str = str(board)
    for piece in board_str:
        if piece == ' ' or piece == '\n':
            continue
        else:
            temp = np.array([[piece_val[piece]]])
            Data_row = np.concatenate((Data_row,temp), axis = 1)
    return Data_row


def next_move(board):
    result = np.empty([0,64], dtype=int)
    for move in board.legal_moves:
        tmp_board = board.copy()
        tmp_board.push(move)
        tmp = board_cvrt(tmp_board)
        result = np.concatenate((result,tmp), axis = 0)
    return result


# Number of moves from board_a to board_b
def nxtmv_count(data_np,board_a, board_b):
    ind = np.where((data_np == board_a).sum(axis = 1) == 64)[0]
    ind = ind+1
    temp = data_np[ind]
    count = ((temp == board_b).sum(axis = 1)== 64).sum()
    return count

def drichlet_count(data_np,board):
    count = np.empty([0,1])
    board_a = board_cvrt(board).reshape([64])
    legal_move = next_move(board)
    for board_b in legal_move:
        tmp = nxtmv_count(data_np,board_a,board_b)
        count = np.concatenate((count,np.array([[tmp]])), axis = 0)
    return count



tmp = data_x.reshape(-1,240,8)
result = np.empty([0,8,8,30])
for i in range(0,tmp.shape[0]):
    temp = tmp[i,:,:]
    temp = temp.reshape(30,8,8).transpose().reshape(1,8,8,30)
    result = np.concatenate((result,temp), axis = 0)
    if i%1000 == 0: print (i)


'''
Number of each type of piece
'''

train_piece_pos = train_h['piece_pos'][:]
test_piece_pos = test_h['piece_pos'][:]

train_h = h5py.File('./DNN/train_data.h5')
test_h = h5py.File('./DNN/test_data.h5')

def piece_num(h5_ptr, start,finish):
    data = h5_ptr['piece_pos'][start:finish]
    result = np.empty((0,12),dtype = 'int8')
    for i in range(data.shape[0]):
        w_p = data[i][0:64].sum()
        w_r = data[i][64:128].sum()
        w_n = data[i][128:192].sum()
        w_b = data[i][192:256].sum()
        w_q = data[i][256:320].sum()
        w_k = data[i][320:384].sum()
        b_p = data[i][384:448].sum()
        b_r = data[i][448:512].sum()
        b_n = data[i][512:576].sum()
        b_b = data[i][576:640].sum()
        b_q = data[i][640:704].sum()
        b_k = data[i][704:768].sum()
        tmp = np.array([[w_p,w_r,w_n,w_b,w_q,w_k,b_p,b_r,b_n,b_b,b_q,b_k]])
        result = np.concatenate((result,tmp), axis = 0)
        if i%10000 == 0:
            # np.save('./Chess/piece_num',result)
            np.save('./Chess/test_piece_num_{}'.format(i), result)
            result = np.empty((0,12),dtype = 'int8')
            print(i)
    np.save('./Chess/test_piece_num_{}'.format(i), result)
    return result


result = np.empty((0,12))
for i in range(0,239):
    temp = np.load('./Chess/train_piece_num_{}.npy'.format(i*10000))
    result = np.concatenate((result,temp), axis = 0)
    print(i)
temp = np.load('./Chess/train_piece_num_2384567.npy')
result = np.concatenate((result,temp), axis = 0)


result = np.empty((0,12))
for i in range(0,103):
    temp = np.load('./Chess/test_piece_num_{}.npy'.format(i*10000))
    result = np.concatenate((result,temp), axis = 0)
    print(i)
temp = np.load('./Chess/test_piece_num_1021957.npy')
result = np.concatenate((result,temp), axis = 0)


data = pd.read_csv('./Chess/pgn_data_titled_2013', index_col = 0)

data_1 = data.iloc[0:25000]
data_2 = data.iloc[25000:41738]

#function equivilant to game_cvrt() but using board_cvrt_sqr() instead
def game_is_check(game):
    board = chess.Board()
    Data_game = pd.DataFrame()
    for moves in game.main_line():
        board.push(moves)
        ck = int(board.is_check())
        ckm = int(board.is_checkmate())
        temp = pd.Series([ck,ckm])
        Data_game = Data_game.append(temp, ignore_index=True)
    return Data_game.astype('int8')



def piece_pos_df(pgn_df):
    temp = pd.DataFrame()
    for i in range(0,pgn_df.size):
        game = chess.pgn.read_game(StringIO(pgn_df.iloc[i,0]))
        temp = temp.append(game_is_check(game), ignore_index=True)
        if i%1000 == 0:
            print(i)
    temp.to_csv('./Chess/board_check_2')
    return temp
