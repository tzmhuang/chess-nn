## Date: 12/04/2018

1. Using updated piece_pos_data to retrain `evl_NN_2_3_2` and `evl_NN_2_4_1` and compare results

2. Using updated piece_pos_data to retrain `evl_NN_2_6` if time allow

3. Adding move_num to training input and us on `evl_NN_2_3_2` or `evl_NN_2_4_1`
    - saved as `evl_NN_3_1`

4. Adding `evl_NN_3_2`
   - Based on `evl_NN_3_1`, but with 10 hidden layers

5. Adding `evl_NN_3_1_2`
   - Based on `evl_NN_3_1` but using MomentumOptimizer

#### results:
1. `evl_NN_2_3_2` and `evl_NN_2_4_1` did not show improvement, they are instead, slight worse.
   - test accuracy are worse.
   - bias term decrease and then levels off
   - check atk_map if they are affected by un-ordered Dict

2. `evl_NN_3_1` result not satisfactory
   - Potential problem: new variable too large, killing gradient
