# Runtime Ensembles

This is a repository for the draft paper: "A Simple Ensmeble Approach for Runtime Monitoring of Neural Networks".

## Getting Started

We use python 3.7 with the pytorch and sklearn packages. Other versions should work as well. To start, set up the datasets and pre trained models (see below),
then run the jupyter notebooks in the main folder containing the experiments.

## Datasets

Please unzip the file in `./data/f_pointnet_eval`. Also, get the OOD datasets from the links below, and place thme in the `./data` floder:

* [Tiny-ImageNet (resize)](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz)
* [LSUN (resize)](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)
* Optional: You can also download the KITTI dataset from http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d and place it inside data/kitti/object.
  This is not needed to run the experiments, since we have included a dict with the needed inputs, it's only optional. To fully reproduce the experiment, after setting    up kitti, go to /f_pointnet_stuff, and run the eval script. 

## Pre trained nets

Get the pre-trained models from the Mahanalobis detector, and place them in models. Densenets are included, you can find the Resnets below:

* [DenseNet on CIFAR-10](https://www.dropbox.com/s/pnbvr16gnpyr1zg/densenet_cifar10.pth?dl=0) / [DenseNet on CIFAR-100](https://www.dropbox.com/s/7ur9qo81u30od36/densenet_cifar100.pth?dl=0) / [DenseNet on SVHN](https://www.dropbox.com/s/9ol1h2tb3xjdpp1/densenet_svhn.pth?dl=0)
* [ResNet on CIFAR-10](https://www.dropbox.com/s/ynidbn7n7ccadog/resnet_cifar10.pth?dl=0) / [ResNet on CIFAR-100](https://www.dropbox.com/s/yzfzf4bwqe4du6w/resnet_cifar100.pth?dl=0) / [ResNet on SVHN](https://www.dropbox.com/s/uvgpgy9pu7s9ps2/resnet_svhn.pth?dl=0)

