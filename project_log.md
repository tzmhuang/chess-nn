## Date: 12/04/2018

1. Using updated piece_pos_data to retrain `evl_NN_2_3_2` and `evl_NN_2_4_1` and compare results

2. Using updated piece_pos_data to retrain `evl_NN_2_6` if time allow

3. Adding move_num to training input and us on `evl_NN_2_3_2` or `evl_NN_2_4_1`
    - saved as `evl_NN_3_1`

4. Adding `evl_NN_3_2`
   - Based on `evl_NN_3_1`
   - but with 10 hidden layers

5. Adding `evl_NN_3_1_2`
   - Based on `evl_NN_3_1`
   - MomentumOptimizer
   - use `one-hot encoding` _(7)_
   - Also trainined `evl_NN_3_1_2_GD` using GradientDescentOptimizer as control

6. Adding `evl_NN_3_1_3`
   - Based on `evl_NN_3_1`
   - scaling move-num by 10e-2
   - using MomentumOptimizer
   - Also training `evl_NN_3_1_3_GD` using GradientDescentOptimizer as control

7. Add create one-hot vector encoding game stage for game phase
   - Opening `n <= 20`
   - Mid-Game `20 < n <= 60`
   - Endgame `n > 60(total_move_num)`

8.  Adding `evl_NN_2_4_mid` and `evl_NN_2_4_late`
   - Seperate mid and late game training


#### results:
1. `evl_NN_2_3_2` and `evl_NN_2_4_1` did not show improvement, they are instead, slight worse.
   - test accuracy are worse.
   - bias term decrease and then levels off
   - check atk_map if they are affected by un-ordered Dict

2. `evl_NN_3_1` result not satisfactory
   - Potential problem: new variable too large, killing gradient

3. `evl_NN_3_1_3` result worse than expected,
   - cost plateau at 1.09 while accuracy hovers below 4.5 but has a steady increasing trend
   - Much less fluctuation in loss comparing to `evl_NN_3_1`
   - higher overall accuracy, but could due to the testing sample selection
   - less training fluctuation in parameter
   - slightly better activation
   - strangly worse distinctive softmax output

4. `evl_NN_3_1_2` result worse than expected
   - `evl_NN_3_1_2_GD` shows potential, with cost reaching 1.05
   - `evl_NN_3_1_2_GD` soft max pattern not as distinctive? as `evl_NN_2_4_1`
5. `evl_NN_2_4_mid` and `evl_NN_2_4_late` does not show any improved performance,
   - Could be due to decrease in training number

#### Conclusion

1. Continue on  `evl_NN_3_1_2_GD`

2. Potential improvement:
   - Adamoptimizer
   - some changes in MomentumOptimizer, usually momentum = 0.9
   - usually minibatch gradientDescent has batch_size = 100-256

3. Try to integrate reinforcement learning

4. Set up google VM GPU to increase training speed


## Date: 14/04/2018

1. Train `evl_NN_3_mini_rand`
   - use minibatch SGD with scattered random data selection
   - first used original rand_batch function with batch size 256
       - training rate: 0.001
       - getting more volatile result, cost lowered to 0.89 but as high as 1.19
   - Using more random approach in selecting data and reduce learning rate back to 0.0001

#### results:
1. `evl_NN_3_mini_rand` with 0.001 learning rate:
   - Learning speed slow
   - immense fluctuation
   - not necessarily better result

2. `evl_NN_3_mini_rand` with 0.0001 learning rate and batch size 512:
   - learning speed/ running speed even slower:
   - result meh..



## Date: 15/04/2018

1. Training `evl_NN_Adam`
   - initial learning rate 0.001
   - batch size 512
   - other hyperparameter using default setup
   - result pretty good, training cost ~0.5
   - training accuracy >0.6
   - interesting soft max pattern
2. Training Training `evl_NN_Adam_2`
   - usning more layers =6
   - same setup
   - extreme weird result
   - works after reducing initial learning rate,
   - no significant improvement 
