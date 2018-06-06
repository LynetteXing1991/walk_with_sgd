# walk_with_sgd
Experiments for the paper "A Walk with SGD" (https://arxiv.org/pdf/1802.08770.pdf)

## Experiment for Figure 2:
1. run sgd to save the model at every iteration, the cosine of angle between consecutive gradients and parameter distance from initialization.

python sgd.py   --data < directory address of data set >   --dataset cifar10   --save_dir < directory address to save model/log files >   --arch vgg11   --lr 0.1   --save_model_per_iter True

2. run the interpolation.

python   interpolation.py   --data < directory address of data set >   --data_set cifar10   --model_dir < same as --save_dir in step 1 >   --arch vgg11   --epoch_index 1 --mode sgd

3. plot the trajectory, cosine of angle between consecutive gradients and the parameter distance from initialization.

python trajectory_plot.py --root_dir < same as --save_dir in step 1 > --epoch_index 1

## Experiment for Figure 1:
1. run gd to save the model at every iteration, the cosine of angle between consecutive gradients and parameter distance from initialization.

python gd.py   --data < directory address of data set >   --dataset cifar10   --save_dir < directory address to save model/log files >   --arch vgg11   --lr 0.1   --save_model_per_iter True

2. run the interpolation

python   interpolation.py   --data < directory address of data set >   --data_set cifar10   --model_dir < same as --save_dir in step 1 >   --arch vgg11   --epoch_index 1 --mode gd

3. plot the trajectory, cosine of angle between consecutive gradients and the parameter distance from initialization.

python trajectory_plot.py --root_dir < same as --save_dir in step 1 > --epoch_index 1

## Experiment for plotting spectral norm of hessian vs epochs:
1. compute the spectral norm of hessian for models saved at desired epochs (by running sgd.py as discussed above).

python spectral_norm.py --data < directory address of data set > --dataset cifar10 --save_dir < same as --save_dir in step 1 of (Experiment for Figure 2) > --epoch 1

The argument '--epoch' can be set to any epoch number for which the spectral norm needs to be computed. Once completed, the path of the saved file is printed on screen.

2. plot the spectral norm vs. epochs.

python spectral_norm.py --files file1_path file2_path ...
The argument --files takes a list (of any length) of file paths printed on screen in step 1 above.


## Options for arguments provided in above experiments:

--dataset: cifar10, mnist

--arch: mlp (only for mnist), vgg11 (only for cifar10), resnet (only for cifar10)


