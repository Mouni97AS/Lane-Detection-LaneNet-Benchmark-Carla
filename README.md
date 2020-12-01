# Lane Detection-LaneNet
Tensorflow implementation of deep neural network for real time lane detection-LaneNet model.


LaneNet architecture consist of encoder-decoder network ENet that modified into two branched network, binary segmentation branch and instance segmentation embedding branch.

![NetWork_Architecture](./data/source_image/network_architecture.png)


### Preliminary Setup
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

### LaneNet-Tusimple Benchmark Dataset
Tusimple dataset release about 7,000 one-second-long video clips of 20 frames each, The advantage of tusimple it provid the files .json the labelled lanes pixels of each image.

The dataset is available [here](https://github.com/TuSimple/tusimple-benchmark/issues/3) move them to the folder data/tusimple
```
Tusimple_path
├── clips
├── label_data_0313.json
├── label_data_0531.json
├── label_data_0601.json
└── test_label.json
```

### Data Preparation
The training folder should contain three folder consist of the orignal image, binary segmentation use 255 for representating the lane, and the instance segmentation use different pixel value to label each lane pixel with a lane Id
run the script:

```
python tools/generate_tusimple_dataset.py --src_dir data/tusimple
```
the script will generate a folder called training contain the three components of orginal rgb image,binary label image, instance label image.  
And process the train.txt where from train.txt we select several lines to obtain the val.txt finally the model ready for train

