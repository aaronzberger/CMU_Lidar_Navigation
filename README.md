# U-Net Training and Testing

Training module for U-Net, including tools for labeling, viewing data, and testing with different hyperparameters.

![Pipeline](https://user-images.githubusercontent.com/35245591/101995864-32835f80-3c9b-11eb-986e-e320d50177a8.png)

## Table of Contents
- [U-Net](#U-Net)
- [Pipeline](#Pipeline)
- [Usage](#Usage)
- [Configuration](#Configuration)
- [Representation](#Representation)
  - [Point Cloud Trimming](#Point-Cloud-Trimming)
  - [Resolution](#Resolution)
- [Labeling](#Labeling)
- [Training Time](#Training-Time)
- [Loss](#Loss)
- [Acknowledgements](#Acknowledgements)

## U-Net
U-Net is a Convolutional Neural Network architecture built for Biomedical Image Segementation (specifically, 
segmentation of neuronal structures in electron microscopic stacks).

> "The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization."

The precise architecture is shown here:

![U-Net Architecture](https://user-images.githubusercontent.com/35245591/101233308-e37b7000-3685-11eb-8318-eedc7b904ef5.png)

The official paper for U-Net can be found [here](https://arxiv.org/abs/1505.04597).

## Pipeline
This module is for the training and testing of the vision node in an autonomous vineyard navigation pipeline.

![Full Pipeline](https://user-images.githubusercontent.com/35245591/101971788-d6b8c800-3c01-11eb-9de2-cea6adaa09b9.jpg)

Code locations for for each node:
- [__VISION__](https://github.com/aaronzberger/CMU_UNet_Node)
- [__ROW TRACKING__](https://github.com/aaronzberger/CMU_EKF_Node)
- __PATH PLANNING__ (Not yet on Github)
- __DRIVING__ (Not yet on Github)

## Usage
There are many executable files in this repository. Except for `main.py` and any configuration files, each file has a multiline
docstring at the top, summarizing the file's purpose and giving its usage command.

For the `main.py` file, which controls training, evalutation, and testing of the network, usage is:

`main.py {train, eval, test}`

## Configuration
This code is built around being able to edit the configuration of the network easily, re-train, re-test, and compare results.

In your base directory, you should have a folder called *experiments*. This will contain your configurations. Currently, there is just one, called *default*. 
The goal is to be able to switch configurations easily, and save results in the folder for whatever configuration you're using.

Each configuration has a `config.json` file. Edit the hyperparameters for the network in this file.

## Representation
It is necessary to convert the raw point clouds into a format that can be input to the network. 

The Birds Eye View (BEV) representation was chosen:

#### Point Cloud Trimming
The point clouds need to be trimmed to ensure all points can fit within the range of the image. This means there must be mins and maxes for X, Y, and Z.

- Width represents the X direction of the robot (forward and backwards)
- Length represents the Y direction of the robot (side to side)
- Height represents the Z direction of the robot (up and down)

These mins and maxes are specified in the `geometry` hyperparameter in your configuration file.

For example, if you wish to only count points ahead of the robot, you may use a width range of (0.0, 10.0). If you wish to use nearly all points ahead and behind the robot, you may use a width range of (-10.0, 10.0).

#### Resolution
Point clouds are received as raw point arrays, where as distance increases, so does the sparsity of points. Since we are representing the point clouds in BEV images, each \[X, Y, Z] pair must be matched to exactly one pixel (and channel). Therefore, there may be two points that map to the same pixel if their Euclidian distance is small enough.

We can therefore see that resolution may affect the performance of the model. Using larger images and more feature channels, we may increase the accuracy of the model by mapping closer points to separate pixels. Keep in mind the size of the image also affects speed and memory usage.

## Labeling
This repository includes an end-to-end labeling pipeline for converting ROS bag files into data that is ready for the network.

Once you have collected bag files that contain point cloud Lidar data, and you have setup your [configuration](#Configuration), you are ready to begin labeling:

1) `bag_extractor.py` will convert your bag files into individual numpy files each containing a point cloud. In the file, provide a directory where your bags are.
2) `run_labeler.py` will create an interactive tool for labeling the data, and save the results in individual numpy files. This will take a while.
3) `view_labels.py` will help you visualize the labels you have created, either in 3D or 2D. You may wish to run this a few times during step 2 to ensure you are making your labels correctly.
4) `split_data.py` will split the data into training and testing datasets and write csv files containing individual paths to the corresponding point cloud files and label files. In the file, provide the split ratio.

At this point, in your base directory, you should have a folder called 'data', which contains:
- *raw*: a folder containing .npz files that have the raw point clouds generated by step 1.
- *labels*: a folder containing .npz files that have the labels generated by step 2.
- *train.csv*, *test.csv*: files containing paths to individual groups of raw and labeled data.

## Training Time
Training on an NVIDIA RTX 2080 with 2,087 point clouds, with a batch size of 4, each epoch takes around 2.9 mins.

With validation after every epoch, full training to 120 epochs will take around 6-8 hours.

## Loss
Currently, the network is trained using only [Classification Loss](https://github.com/aaronzberger/CMU_Lidar_Navigation/blob/main/srcs/loss.py) (binary cross-entropy).

However, in the future, I hope to add Embedding Loss, which attempts to increase the separation between classes (in this case, vine or no vine) by applying constrastive loss or center loss to the feature space or the classifier space. This has been shown to decrease the difference between training and testing Classification Loss for aerial images.

For details, see [this paper](https://arxiv.org/pdf/1712.01511.pdf).

## Acknowledgements
- John Macdonald for Lidar pre-processing, most of the labeling pipeline, and more
- Olaf Ronneberger, Philipp Fischer, Thomas Brox for the [U-Net Architecture](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- Jon Binney for [`numpy_pc2.py`](https://github.com/dimatura/pypcd/blob/master/pypcd/numpy_pc2.py)
