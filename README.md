# rcnn_for_predicting_wrong_pics
This is a project that USES RCNN and Faster-RCNN to find the wrong pictures of VOC dataset target detection

注意整个项目的代码架构在我的博客中解释了。[欢迎大家关注](https://www.cnblogs.com/ginkgo-/p/13838833.html)

Note that the code architecture for the entire project is explained in my blog.You are welcome to pay attention.

faster-rcnn文件夹里面包含的是借助于[simple-faster-rcnn](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)完成的利用faster-rcnn检测预测错误的图片。其他的为RCNN需要的文件。

The faster-rcnn folder contains images that detect prediction errors using the use of fast-rcnn with the help of the use of simple-faster-rcnn.Others are for the RCNN.

./model/checkpoints文件夹里面应该包含的模型的所有权重，如果想要获取，[点击这儿，提取码为pfms](https://pan.baidu.com/s/1rCRXDiR_41KjEA9rBYyHYg)

All the weights of the model that should be included in the ./model/checkpoints folder, if you want to get them, click here, [the extraction code is pfms](https://pan.baidu.com/s/1rCRXDiR_41KjEA9rBYyHYg)

./VOCdevkit文件夹是用于保存VOC2007数据集，可以采用下面方式获取：

The ./VOCdevkit folder is used to hold the VOC2007 dataset and can be obtained as:

```shell
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```
