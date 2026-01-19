# COMPUTER_VISION_PEPSI

In order to start the training process for the model use the following command:
py -m train --model partial_conv
switching between the various possible models (pepsi_pp, pepsi, lama_like and partial_conv)

This will produce various folders containing the checkpoints of the models, by default there will be 
150 epochs and the checkpoints will be saved every 50, there will also be some sample validation
for the models on such epochs

After doing so is possible to call the evaluation process by using: 
py -m evaluation --ckpt .\checkpoints\ckpt_partial_conv_epoch150.pth --model partial_conv --dataset .\data\val_data
here aswell will be possible to switch between the various models, is also important to remember to specify 
always the correct checkpoint path to evaluate the model

At last the user will have the possibility to evaluate the FID directly launching the evaluation 
from command line:
pytorch-fid .\eval_out_partial_conv\orig .\eval_out_partial_conv\generated
(still switching between the various models)
