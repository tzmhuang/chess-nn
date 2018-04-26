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


## Date: 15/04/2018

1. Using evl_Adam for chess engine evaluation
   - Differentiable probability output in very early game
   - All state win probability goes to 1

2. Training `evl_Adam_AE`
   - Using auto encoder,
   - Gradually decrease layer size: [1544,1000,500,3]
   - removing game_phase variable
   - batch size = 1024
   - others same as evl_NN_Adam
   - Could try autoencoder + auto decoder
   - training loss minimized to 0.2 and can go lower
   - testing accuracy no improvement
   - softmax output extreme, 0 and 1s, little in between

3. Training `evl_Adam_L2reg`
   - based on `evl_Adam_AE`
   - adding L2_regularization
   - beta 0.01 () could be too large
      - 0.01 training convergence low, but testing accuracy is better
      - 0.001, training loss decreases, testing accuracy also decreases, tops at around 4.7
   - trying to add back piece_val, but with scaling (x-min)/(max-min)

4. Re-arrange atk_map
   - Detecting problem with atk_map
   - re-run data

5. retrain `evl_Adam_L2reg` dropping atk_map data
   -

## Date: 21/04/2017

1. Batch Normalization

2. how do tf.layer.conv2d initialize

3. ReLu naturally enforce sparsity?

4. Better loss function?

5. Adam? is it required

## Date: 23/04/2017

1. Installed GPU on instance1
   - installed `CUDO 9.1` with `cuNN 7.1.3`
   - rebuild tensorflow from bazel with gpu implementation
      - see http://www.python36.com/install-tensorflow141-gpu/
      - process parallel over CPU
      - 4cpu takes 5000+ sec
      - DONE

2. trying training on GPU using `evl_conv_temp`
   - Monitor GPU usage:
      - `nvidia-smi`
   - Using GPU Success:
      - Speed increased by 10x
   - accuracy = 0.4227669, global_step = 23287, loss = 1.0676591

3. Trainnig `evl_conv_1`
   - training on
      - `NVIDIA K80` + 4 CPU
   - AdamOptimizer
   - 3 conv layer with 128 filter, 2 dense layer,
   - adding single filter conv layer before dense
   - batch size 512
   - accuracy = 0.4227669, global_step = 23287, loss = 1.0730845


4. Trainnig `evl_conv_2`
   - using `GradientDescentOptimizer`
   - training on
      - `NVIDIA K80` + 4 CPU
   - 3 conv layer with 128 filter, 2 dense layer,
   - adding single filter conv layer before dense
   - batch size 512
   - accuracy = 0.42275614, global_step = 23287, loss = 1.0645185

5. Re-train `evl_conv_1`
   - using 0.0001 as initialized AdamOptimizer
   - batch_size = 1024
   - accuracy = 0.47654992, global_step = 11644, loss = 1.0147022

6. Trainnig `evl_conv_2`
   - using `AdamOptimizer` w 0.0001
   - training on
      - `NVIDIA K80` + 4 CPU
   - 3 conv layer with 128 filter, 2 dense layer,
   - With batch normalization
   - adding single filter conv layer before dense
   - batch size 1024
   - accuracy = 0.47183543, global_step = 11645, loss = 1.0219235
