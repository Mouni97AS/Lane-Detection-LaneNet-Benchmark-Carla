# Lane Detection-LaneNet
Tensorflow implementation of deep neural network for real time lane detection-LaneNet model
LaneNet architecture consist of encoder-decoder network ENet that modified into two branched network, binary segmentation branch and instance segmentation embedding branch.

![NetWork_Architecture](./data/source_image/network_architecture.png).


##Preliminary Setup
The project was implement on IDE Pycharm by configuring a conda environment of tensorflow 1.14 and the requirment below:
### Requirements 
- python3.7
- numpy
- tqdm
- glog
- easydict
- tensorflow_gpu
- matplotlib
- opencv
- scikit_learn
- loguru
