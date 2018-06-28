### Description

Given a picture with a bird, we are supposed to box the bird.

<div align=center><img  src="https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/before.png"/></div>

![](https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/before.png)

In src/data directory, ```images.txt``` is the index of all images, ```bouding_boxex.txt``` is the label box of all images and  ```images``` contains all images. Box data make up of 4 data: the top left corner coordinate of box, width of box and height of box. 



### Neural Network

For traditional CNN and FC, it will meet degeneration problems when layers go deep.

![](https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/degeneration1.png)

![](https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/degeneration2.png)

In paper ```Deep Residual Learning for Image Recognition```, they try to solve this problem by using a Residual Block:

![](https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/ResidualBlock.png)

These blocks compose ResNet:

![](https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/ResNet.png)

I use ResNet-18 in this project by adding a 4-dimension layer after ResNet-18 to predict box's x, y ,w and h.

Loss: smooth l1 loss

![](https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/loss.png)

Metric: IoU of groound truth and prediction, threshold=0.75

![](https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/IoU.png)



### Train

Resize all images to ```224*224*3```

![](https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/after.png)

Then normalize and standardize all pixel channel.

Split all data into 9000 training data and 2788 tesing data. Train network on training data using ```batch size=128```, ```epoch=100``` and ```validation split ratio=0.1```

Training result:

![](https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/model.png)

Testing result:

![](https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/test.png)



### Examples

Red box represents ground truth and green box is the prediction of network.

![](https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/result1.png)

![](https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/result2.png)

![](https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/result3.png)

![](https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/result4.png)

![](https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/result5.png)

![](https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/result6.png)

Failed example:

![](https://github.com/CKCZZJ/Image-Object-Localization/blob/master/img/fail.png)



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

[]: https://arxiv.org/pdf/1512.03385.pdf	"Deep Residual Learning for Image Recognition"



### Author

CKCZZJ



### Licence 

MIT