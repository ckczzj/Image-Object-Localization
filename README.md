### Description

Given a picture with a bird, we are supposed to box the bird.

<div align=center><img  src="https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/before.png"/></div>

In src/data directory, ```images.txt``` is the index of all images, ```bouding_boxex.txt``` is the label box of all images and  ```images``` contains all images. Box data make up of 4 data: the top left corner coordinate of box, width of box and height of box. 



### Neural Network

For traditional CNN and FC, it will meet degeneration problems when layers go deep.

<figure class="half">
<img  src="https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/degeneration1.png"/>

<img  src="https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/degeneration2.png"/>
</figure>

In paper ```Deep Residual Learning for Image Recognition```, they try to solve this problem by using a Residual Block:

<div align=center><img  src="https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/ResidualBlock.png"/></div>

These blocks compose ResNet:

<div align=center><img  src="https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/ResNet.png"/></div>

I use ResNet-18 in this project by adding a 4-dimension layer after ResNet-18 to predict box's x, y ,w and h.

Loss: smooth l1 loss

<div align=center><img  src="https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/loss.png"/></div>

Metric: IoU of groound truth and prediction, threshold=0.75

<div align=center><img  src="https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/IoU.png"/></div>



### Train

Resize all images to ```224*224*3```

<div align=center><img  src="https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/after.png"/></div>

Then normalize and standardize all pixel channel.

Split all data into 9000 training data and 2788 tesing data. Train network on training data using ```batch size=128```, ```epoch=100``` and ```validation split ratio=0.1```

Training result:

<div align=center><img  src="https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/model.png"/></div>

Testing result:

<div align=center><img  src="https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/test.png"/></div>



### Examples

Red box represents ground truth and green box is the prediction of network.

<div align=center><img  src="https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/result1.png"/></div>
<div align=center><img  src="https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/result2.png"/></div>
<div align=center><img  src="https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/result3.png"/></div>
<div align=center><img  src="https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/result4.png"/></div>
<div align=center><img  src="https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/result5.png"/></div>
<div align=center><img  src="https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/result6.png"/></div>

Failed example:

<div align=center><img  src="https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/fail.png"/></div>



### Attention

You should keep the directory structure.



### Dependency

```src```

python 3.6

tensorflow 1.3.0

keras 2.1.0

numpy

PIL

pickle

matplotlib



### Run

In ```src``` directory:

```python getdata.py``` to preprocess data.

If you want to train model, ```python train.py```

If you want to test on trained model(if you had trained model), ```python test.py```



### Reference

[Deep Residual Learning for Image Recognition]: https://arxiv.org/pdf/1512.03385.pdf



### Author

CKCZZJ



### Licence 

MIT