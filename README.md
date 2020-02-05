# A Deep Chess Engine with Minimax Forward Mining


## Description

This project aims to develop a chess engine (Engine) that uses deep learning and Bayesian theorem for chess board evaluation and minimax algorithm with alpha beta pruning for game tree searching. The project is written in python and the deep learning models are built with Tensorflow.

Two neural network models ANN and ConvNN are built for board evaluation. The prediction results were compared, and the ANN model was selected to be used in the final Engine due to its superior performance. Applying Bayesian theorem to board evaluation does show preliminary success, as the engine would perform more reliable opening moves. However more thorough testing is needed for a more conclusive performance evaluation.

The Engine was evaluated with existing chess strategy test suits to test its understandings of different strategic themes. It also played against an existing open source chess engine to test its general playing strength. The overall ability of the Engine is un satisfactory, but it could be a reason of limited search depth due to insufficient computing power. Again, more testing is required for a more comprehensive assessment.

## Data

`<ficsgamesdb_2013_titled.pgn>`

FICS games 2013 with titled player only. In total 41738 games

## Extracted Data

`chess_data.h5` 

keys(): 

- `atk_map` 
   - 8x8 bitmap encoding attack map for each piece.
   - shape: [3406526,768]
- `board_set`
   - with
   `{'P':1, 'R':5, 'N':3, 'B':4, 'Q':9, 'K':100,'p':-1, 'r':-5, 'n':-3, 'b':-4, 'q':-9, 'k':-100,'.':0}`
   - shape: [3406526,64]
- `castling`
   - castling flag
   - shape: [3406526,4]
- `game_move_num`
   - the game number and move number
   - shape: [3406526,2]
- `piece_pos`
   - 8x8 bitmap encoding for every piece type
   - shape: [3406526,768]
- `result`
   - Flag showing result of game
    `1 for win`, `0 for draw`, `-1 for lose`
   - shape: [3406526,1]
- `turn_move`
   - moveing side flag
   `0 for white 1 for black`
- `flag`
   - one-hot encoding for 
   `result`

## Result

The result shows that the Engine performs rather poorly. However, as it showed a fair performance in board state evaluation in games played against Sunfish, it is likely that the Engine is limited by its search algorithm. Originally this project hopes to use a probabilistic tree pruning method to limit the nodes searched and reach a deeper search depth, however there is not enough time.

In the future, I hope to develop a more efficient search algorithm and a better pruning method to decrease search time. Moreover, I wish to incorporate the project with reinforcement learning to achieve an even better performance.


