# chess-nn

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
