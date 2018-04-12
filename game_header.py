import chess
import numpy as np
import pandas as pd
from io import StringIO

'''
file directory: "Desktop/Chess/ficsgamesdb_2013_titled.pgn"
file directory: "/Desktop/Chess/tmp.pgn"

'''

def pgn_file_reader(file_dir):
    pgn = open(file_dir)
    line_br_detector = 0
    pgn_data_frame = pd.DataFrame()
    #temp0 = pd.Series()
    game_text = ""
    temp1 = pd.Series()
    counter = 0
    for line in pgn.read().splitlines():                                        #Forming DataFrame with Pandas
        game_text = game_text + line + '\n'
        if line == "":
            line_br_detector += 1
            game_text = game_text + "" +'\n'
        if line_br_detector == 1:
            continue
        elif line_br_detector > 0 and line_br_detector%2 == 0:
            temp1 = pd.Series(game_text)
            pgn_data_frame = pgn_data_frame.append(temp1, ignore_index=True)
            temp1 = pd.Series()
            game_text = ""
            line_br_detector = 0
            counter += 1
            if counter%5000 == 0:
                print (counter)
            continue
    pgn.close()
    return pgn_data_frame


'''
pgn_text parsing codes
for some i in [0,pgn_data.size[0]]
'''
pgn_string = pgn_df.iloc[i,0]
pgn = StringIO(pgn_string)
game = chess.pgn.read_game(pgn)
game.headers
