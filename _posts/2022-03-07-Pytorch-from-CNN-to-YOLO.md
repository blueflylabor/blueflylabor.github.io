---
layout: post
title:  "Pytorch from CNN to YOLO"
date:   2022-03-07
last_modified_at: 2022-05-07
categories: [Pytorch, CNN, YOLO]
---
![cnn1](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/cnn1.png)

![cnn2 (2)](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/cnn2%20(2).png)

![cnn3](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/cnn3.png)

**分类猫和狗**

使用一个还不错的相机采集图片(12M)   

RGB figure 36M 元素  

使用100大小的单隐藏层MLP 模型有3.6B = 14GB 元素   

远多于世界上所有的猫狗总数(900M dog 600M cat)  

**两个原则**

平移不变性  

局部性  

**重新考察全连接层**  

将输入和输出变形为矩阵（宽度，高度）

将权重变形为4-D张量（h,w）到（h',w'）
$$
h_{i,j}=\sum_{k,l}w_{i,j,k,l}x_{k,l}=\sum_{a,b}=v_{i,j,a,b}x_{i+a,j+b}
$$
V是W的重新索引
$$
v_{i,j,a,b}=w_{i,j,i+a,j+b}
$$


**原则#1 - 平移不变性**

x的平移导致h的平移
$$
h_{i,j}=\sum_{a,b}v_{i,j,a,b}x_{i+a,j+b}
$$
v不应依赖于（i, j）  

解决方案：
$$
v_{i,j,a,b}=v_{a, b},
h_{i,j}=\sum_{a,b}v_{a,b}x_{i+a,j+b}
$$
这就是交叉相关  

**原则#2 - 局部性**

### 局部性


$$
\begin{aligned}
&为了收集用来训练参数[\mathbf{H}]_{i, j}的相关信息，\\
&我们不应偏离到距(i, j)很远的地方。\\
&这意味着在|a|> \Delta或|b| > \Delta的范围之外，\\
&我们可以设置[\mathbf{V}]_{a, b} = 0。\\
&因此，我们可以将[\mathbf{H}]_{i, j}重写为:\\
&[\mathbf{H}]*_{i, j} = u + \sum_*{a = -\Delta}^{\Delta} \sum*_{b = -\Delta}^{\Delta} [\mathbf{V}]_*{a, b} [\mathbf{X}]_{i+a, j+b}.
\end{aligned}
$$

当图像处理的局部区域很小时，卷积神经网络与多层感知机的训练差异可能是巨大的：以前，多层感知机可能需要数十亿个参数来表示网络中的一层，而现在卷积神经网络通常只需要几百个参数，而且不需要改变输入或隐藏表示的维数。

参数大幅减少的代价是，我们的特征现在是平移不变的，并且当确定每个隐藏活性值时，每一层只包含局部的信息。

以上所有的权重学习都将依赖于归纳偏置。当这种偏置与现实相符时，我们就能得到样本有效的模型，并且这些模型能很好地泛化到未知数据中。

但如果这偏置与现实不符时，比如当图像不满足平移不变时，我们的模型可能难以拟合我们的训练数据。

![cnn4](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/cnn4.png)

![image-20220127104222384](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/cnn5.png)

![cnn6](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/cnn6.png)

![cnn7](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/cnn7.png)

![cnn8](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/cnn8.png)

![cnn9](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/cnn9.png)

![image-20220127105601246](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/image-20220127105601246.png)

## Sharing-Weight

![cnn11](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/cnn11.png)

![image-20220127110147649](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/image-20220127110147649.png)

![cnn12](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/cnn12.png)

![cnn13](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/cnn13.png)

![cnn14](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/cnn14.png)

![cnn15](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/cnn15.png)

![cnn16](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/cnn16.png)

![cnn17](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/cnn17.png)

![cnn18](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/cnn18.png)

![cnn19](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/cnn19.png)

## Pooling - Max Pooling

![cnn20](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/cnn20.png)

![cnn20.1](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/cnn20.1.png)

**Max-Pooling:选取最大的值 也可选取其他的采用 当然也可不做采用前提是性能足够**

![cnn21](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/cnn21.png)

![cnn22](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/cnn22.png)

![cnn23](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/cnn23.png)

**但CNN无法直接对一个放大的图像做识别，需要data augmentation(对数据集进行旋转，放大，缩小，等操作)**

# YOLOv1

## Bounding-Box

将一张图片分割为有限个单元格(Cell,图中红色网格)   
![split-pic](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/split-image.png)  
每一个输出和标签都是针对每一个单元格的物体中心(midpiont,图中蓝色圆点)
每一个单元格会有[X1, Y1, X2, Y2]
对应的物体中心会有一个[X, Y, W, H]  
![bb1](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/bounding-box1.png)
X, Y 在[0, 1]内表示水平或垂直的距离  
W, H > 1 表示物体水平或垂直方向上高于该单元格 数值表示水平或垂直方向的单位长度的倍数  
[0.95, 0.55, 0.5, 1.5]=>显然图像靠近右下角 单元格不能表示出完整的物体  
![bb2](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/bounding-box2.png)
根据 [X, Y, W, H] => [0.95, 0.55, 0.5, 1.5] 计算得到Bounding Box(图中蓝色网格)

![bbx3](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/b-box-seq.png)

## Image-Label

$$
\begin{aligned}
&label_{cell}=[C_1,C_2,\cdots,C_{20},P_c,X,Y,W,H]\\
&[C_1,C_2,\cdots,C_{20}]:20\space different\space classes\\
&[P_c]:Probability\space for\space there\space is\space an\space object(0\or1)\\
&[X,Y,W,H]:Bounding-Box\\
&pred_{cell}=[C_1,C_2,\cdots,C_{20},P_{c1},X_1,Y_1,W_1,H_1,P_{c2},X_2,Y_2,W_2,H_2]\\
&Taget\space shape\space for\space one \space images:(S, S, 25)\\
&Predication\space shape \space for\space one\space images:(S,S,30)\\
\end{aligned}
$$

## Model-Framework

![yolov1](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/yolov1-modelfw.png)

## Pytorch implement

```python
from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
```


```python
# remove same elements in tensor 
def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

```


```python
a = torch.tensor([1., 2., 3., 1., 3.,])
a, unique(a)
```




    (tensor([1., 2., 3., 1., 3.]), tensor([1., 2., 3.]))




```python
def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou
```


```python

```


```python
def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):

    
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride
    
    return prediction
```


```python
def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = prediction.size(0)

    write = False
    


    for ind in range(batch_size):
        image_pred = prediction[ind]          #image Tensor
       #confidence threshholding 
       #NMS
    
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        if image_pred_.shape[0] == 0:
            continue       
#        
  
        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1])  # -1 index holds the class index
        
        
        for cls in img_classes:
            #perform NMS

        
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections
            
            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at 
                #in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
            
                except IndexError:
                    break
            
                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
            
                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))

    try:
        return output
    except:
        return 0
```


```python
def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas
```


```python
def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img
```


```python
def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

```


```python
from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
```


```python
def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_
```


```python
get_test_input()
```




    tensor([[[[0.2392, 0.2392, 0.2392,  ..., 0.8431, 0.5373, 0.2627],
              [0.2392, 0.2392, 0.2392,  ..., 0.8078, 0.4706, 0.2353],
              [0.2392, 0.2392, 0.2392,  ..., 0.7804, 0.3843, 0.2510],
              ...,
              [0.6275, 0.6275, 0.6235,  ..., 0.5137, 0.3922, 0.2431],
              [0.6275, 0.6275, 0.6235,  ..., 0.5059, 0.3490, 0.2118],
              [0.6235, 0.6235, 0.6235,  ..., 0.4980, 0.3216, 0.1922]],
    
             [[0.2392, 0.2392, 0.2392,  ..., 0.9294, 0.5647, 0.2667],
              [0.2392, 0.2392, 0.2392,  ..., 0.8941, 0.4941, 0.2314],
              [0.2392, 0.2392, 0.2392,  ..., 0.8627, 0.4078, 0.2510],
              ...,
              [0.6627, 0.6627, 0.6588,  ..., 0.4980, 0.3765, 0.2275],
              [0.6627, 0.6627, 0.6588,  ..., 0.4902, 0.3333, 0.1961],
              [0.6588, 0.6588, 0.6588,  ..., 0.4824, 0.3059, 0.1765]],
    
             [[0.2235, 0.2235, 0.2235,  ..., 0.6039, 0.3294, 0.1098],
              [0.2235, 0.2235, 0.2235,  ..., 0.5843, 0.2745, 0.0902],
              [0.2235, 0.2235, 0.2235,  ..., 0.5804, 0.2078, 0.1176],
              ...,
              [0.7137, 0.7137, 0.7098,  ..., 0.4824, 0.3647, 0.2157],
              [0.7137, 0.7137, 0.7098,  ..., 0.4784, 0.3216, 0.1843],
              [0.7098, 0.7098, 0.7098,  ..., 0.4706, 0.2902, 0.1686]]]])




```python
def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces
    
    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == "[":               # This marks the start of a new block
            if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)     # add it the blocks list
                block = {}               # re-init the block
            block["type"] = line[1:-1].rstrip()     
        else:
            key,value = line.split("=") 
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    
    return blocks
```


```python
blocks = parse_cfg('cfg/yolov3.cfg')
l = len(blocks)
blocks[l-1], blocks[l-1]['type']
```




    ({'type': 'yolo',
      'mask': '0,1,2',
      'anchors': '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326',
      'classes': '80',
      'num': '9',
      'jitter': '.3',
      'ignore_thresh': '.5',
      'truth_thresh': '1',
      'random': '1'},
     'yolo')



我们的函数将会返回一个 nn.ModuleList。这个类几乎等同于一个包含 nn.Module 对象的普通列表。然而，当添加 nn.ModuleList 作为 nn.Module 对象的一个成员时（即当我们添加模块到我们的网络时），所有 nn.ModuleList 内部的 nn.Module 对象（模块）的 parameter 也被添加作为 nn.Module 对象（即我们的网络，添加 nn.ModuleList 作为其成员）的 parameter。

当我们定义一个新的卷积层时，我们必须定义它的卷积核维度。虽然卷积核的高度和宽度由 cfg 文件提供，但卷积核的深度是由上一层的卷积核数量（或特征图深度）决定的。这意味着我们需要持续追踪被应用卷积层的卷积核数量。我们使用变量 prev_filter 来做这件事。我们将其初始化为 3，因为图像有对应 RGB 通道的 3 个通道。

路由层（route layer）从前面层得到特征图（可能是拼接的）。如果在路由层之后有一个卷积层，那么卷积核将被应用到前面层的特征图上，精确来说是路由层得到的特征图。因此，我们不仅需要追踪前一层的卷积核数量，还需要追踪之前每个层。随着不断地迭代，我们将每个模块的输出卷积核数量添加到 output_filters 列表上。

现在，一个空的层可能会令人困惑，因为它没有做任何事情。而 Route Layer 正如其它层将执行某种操作（获取之前层的拼接）。在 PyTorch 中，当我们定义了一个新的层，我们在子类 nn.Module 中写入层在 nn.Module 对象的 forward 函数的运算。

对于在 Route 模块中设计一个层，我们必须建立一个 nn.Module 对象，其作为 layers 的成员被初始化。然后，我们可以写下代码，将 forward 函数中的特征图拼接起来并向前馈送。最后，我们执行网络的某个 forward 函数的这个层。

但拼接操作的代码相当地短和简单（在特征图上调用 torch.cat），像上述过程那样设计一个层将导致不必要的抽象，增加样板代码。取而代之，我们可以将一个假的层置于之前提出的路由层的位置上，然后直接在代表 darknet 的 nn.Module 对象的 forward 函数中执行拼接运算。（如果感到困惑，我建议你读一下 nn.Module 类在 PyTorch 中的使用）。


```python
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors       

def create_modules(blocks):
    net_info = blocks[0]     #Captures the information about the input and pre-processing    
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential() # calculate graph seq
    
        #check the type of block
        #create a new module for the block
        #append to module_list
        
        #If it's a convolutional layer
        if (x["type"] == "convolutional"):
            #Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
        
            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
        
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
        
            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)
        
            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
        
            #Check the activation. 
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)
        
            #If it's an upsampling layer
            #We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{}".format(index), upsample)
                
        #If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            #Start  of a route
            start = int(x["layers"][0])
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            #Positive anotation
            if start > 0: 
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]
    
        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
            
        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
    
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
    
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
                              
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        
    return (net_info, module_list)
```


```python
route = EmptyLayer()
blocks = parse_cfg("cfg/yolov3.cfg")
blocks, create_modules(blocks)
```




    ([{'type': 'net',
       'batch': '1',
       'subdivisions': '1',
       'width': '416',
       'height': '416',
       'channels': '3',
       'momentum': '0.9',
       'decay': '0.0005',
       'angle': '0',
       'saturation': '1.5',
       'exposure': '1.5',
       'hue': '.1',
       'learning_rate': '0.001',
       'burn_in': '1000',
       'max_batches': '500200',
       'policy': 'steps',
       'steps': '400000,450000',
       'scales': '.1,.1'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '32',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '64',
       'size': '3',
       'stride': '2',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '32',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '64',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'shortcut', 'from': '-3', 'activation': 'linear'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '128',
       'size': '3',
       'stride': '2',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '64',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '128',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'shortcut', 'from': '-3', 'activation': 'linear'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '64',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '128',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'shortcut', 'from': '-3', 'activation': 'linear'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '256',
       'size': '3',
       'stride': '2',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '128',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '256',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'shortcut', 'from': '-3', 'activation': 'linear'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '128',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '256',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'shortcut', 'from': '-3', 'activation': 'linear'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '128',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '256',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'shortcut', 'from': '-3', 'activation': 'linear'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '128',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '256',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'shortcut', 'from': '-3', 'activation': 'linear'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '128',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '256',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'shortcut', 'from': '-3', 'activation': 'linear'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '128',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '256',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'shortcut', 'from': '-3', 'activation': 'linear'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '128',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '256',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'shortcut', 'from': '-3', 'activation': 'linear'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '128',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '256',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'shortcut', 'from': '-3', 'activation': 'linear'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '512',
       'size': '3',
       'stride': '2',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '256',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '512',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'shortcut', 'from': '-3', 'activation': 'linear'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '256',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '512',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'shortcut', 'from': '-3', 'activation': 'linear'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '256',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '512',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'shortcut', 'from': '-3', 'activation': 'linear'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '256',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '512',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'shortcut', 'from': '-3', 'activation': 'linear'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '256',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '512',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'shortcut', 'from': '-3', 'activation': 'linear'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '256',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '512',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'shortcut', 'from': '-3', 'activation': 'linear'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '256',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '512',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'shortcut', 'from': '-3', 'activation': 'linear'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '256',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '512',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'shortcut', 'from': '-3', 'activation': 'linear'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '1024',
       'size': '3',
       'stride': '2',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '512',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '1024',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'shortcut', 'from': '-3', 'activation': 'linear'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '512',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '1024',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'shortcut', 'from': '-3', 'activation': 'linear'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '512',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '1024',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'shortcut', 'from': '-3', 'activation': 'linear'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '512',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '1024',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'shortcut', 'from': '-3', 'activation': 'linear'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '512',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'filters': '1024',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '512',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'filters': '1024',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '512',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'filters': '1024',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'filters': '255',
       'activation': 'linear'},
      {'type': 'yolo',
       'mask': '6,7,8',
       'anchors': '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326',
       'classes': '80',
       'num': '9',
       'jitter': '.3',
       'ignore_thresh': '.5',
       'truth_thresh': '1',
       'random': '1'},
      {'type': 'route', 'layers': ['-4']},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '256',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'upsample', 'stride': '2'},
      {'type': 'route', 'layers': ['-1', ' 61']},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '256',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'filters': '512',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '256',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'filters': '512',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '256',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'filters': '512',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'filters': '255',
       'activation': 'linear'},
      {'type': 'yolo',
       'mask': '3,4,5',
       'anchors': '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326',
       'classes': '80',
       'num': '9',
       'jitter': '.3',
       'ignore_thresh': '.5',
       'truth_thresh': '1',
       'random': '1'},
      {'type': 'route', 'layers': ['-4']},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '128',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'upsample', 'stride': '2'},
      {'type': 'route', 'layers': ['-1', ' 36']},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '128',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'filters': '256',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '128',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'filters': '256',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'filters': '128',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'batch_normalize': '1',
       'size': '3',
       'stride': '1',
       'pad': '1',
       'filters': '256',
       'activation': 'leaky'},
      {'type': 'convolutional',
       'size': '1',
       'stride': '1',
       'pad': '1',
       'filters': '255',
       'activation': 'linear'},
      {'type': 'yolo',
       'mask': '0,1,2',
       'anchors': '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326',
       'classes': '80',
       'num': '9',
       'jitter': '.3',
       'ignore_thresh': '.5',
       'truth_thresh': '1',
       'random': '1'}],
     ({'type': 'net',
       'batch': '1',
       'subdivisions': '1',
       'width': '416',
       'height': '416',
       'channels': '3',
       'momentum': '0.9',
       'decay': '0.0005',
       'angle': '0',
       'saturation': '1.5',
       'exposure': '1.5',
       'hue': '.1',
       'learning_rate': '0.001',
       'burn_in': '1000',
       'max_batches': '500200',
       'policy': 'steps',
       'steps': '400000,450000',
       'scales': '.1,.1'},
      ModuleList(
        (0): Sequential(
          (conv_0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_0): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_0): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (1): Sequential(
          (conv_1): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (batch_norm_1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_1): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (2): Sequential(
          (conv_2): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_2): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (3): Sequential(
          (conv_3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_3): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (4): Sequential(
          (shortcut_4): EmptyLayer()
        )
        (5): Sequential(
          (conv_5): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (batch_norm_5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_5): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (6): Sequential(
          (conv_6): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_6): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_6): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (7): Sequential(
          (conv_7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_7): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (8): Sequential(
          (shortcut_8): EmptyLayer()
        )
        (9): Sequential(
          (conv_9): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_9): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_9): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (10): Sequential(
          (conv_10): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_10): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_10): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (11): Sequential(
          (shortcut_11): EmptyLayer()
        )
        (12): Sequential(
          (conv_12): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (batch_norm_12): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_12): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (13): Sequential(
          (conv_13): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_13): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_13): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (14): Sequential(
          (conv_14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_14): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_14): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (15): Sequential(
          (shortcut_15): EmptyLayer()
        )
        (16): Sequential(
          (conv_16): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_16): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_16): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (17): Sequential(
          (conv_17): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_17): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_17): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (18): Sequential(
          (shortcut_18): EmptyLayer()
        )
        (19): Sequential(
          (conv_19): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_19): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_19): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (20): Sequential(
          (conv_20): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_20): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_20): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (21): Sequential(
          (shortcut_21): EmptyLayer()
        )
        (22): Sequential(
          (conv_22): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_22): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_22): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (23): Sequential(
          (conv_23): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_23): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_23): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (24): Sequential(
          (shortcut_24): EmptyLayer()
        )
        (25): Sequential(
          (conv_25): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_25): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_25): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (26): Sequential(
          (conv_26): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_26): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_26): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (27): Sequential(
          (shortcut_27): EmptyLayer()
        )
        (28): Sequential(
          (conv_28): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_28): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_28): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (29): Sequential(
          (conv_29): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_29): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_29): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (30): Sequential(
          (shortcut_30): EmptyLayer()
        )
        (31): Sequential(
          (conv_31): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_31): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_31): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (32): Sequential(
          (conv_32): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_32): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_32): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (33): Sequential(
          (shortcut_33): EmptyLayer()
        )
        (34): Sequential(
          (conv_34): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_34): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_34): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (35): Sequential(
          (conv_35): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_35): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_35): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (36): Sequential(
          (shortcut_36): EmptyLayer()
        )
        (37): Sequential(
          (conv_37): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (batch_norm_37): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_37): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (38): Sequential(
          (conv_38): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_38): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_38): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (39): Sequential(
          (conv_39): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_39): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_39): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (40): Sequential(
          (shortcut_40): EmptyLayer()
        )
        (41): Sequential(
          (conv_41): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_41): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_41): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (42): Sequential(
          (conv_42): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_42): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_42): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (43): Sequential(
          (shortcut_43): EmptyLayer()
        )
        (44): Sequential(
          (conv_44): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_44): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_44): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (45): Sequential(
          (conv_45): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_45): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_45): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (46): Sequential(
          (shortcut_46): EmptyLayer()
        )
        (47): Sequential(
          (conv_47): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_47): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_47): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (48): Sequential(
          (conv_48): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_48): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_48): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (49): Sequential(
          (shortcut_49): EmptyLayer()
        )
        (50): Sequential(
          (conv_50): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_50): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_50): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (51): Sequential(
          (conv_51): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_51): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_51): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (52): Sequential(
          (shortcut_52): EmptyLayer()
        )
        (53): Sequential(
          (conv_53): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_53): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_53): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (54): Sequential(
          (conv_54): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_54): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_54): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (55): Sequential(
          (shortcut_55): EmptyLayer()
        )
        (56): Sequential(
          (conv_56): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_56): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_56): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (57): Sequential(
          (conv_57): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_57): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_57): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (58): Sequential(
          (shortcut_58): EmptyLayer()
        )
        (59): Sequential(
          (conv_59): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_59): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_59): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (60): Sequential(
          (conv_60): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_60): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_60): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (61): Sequential(
          (shortcut_61): EmptyLayer()
        )
        (62): Sequential(
          (conv_62): Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (batch_norm_62): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_62): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (63): Sequential(
          (conv_63): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_63): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_63): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (64): Sequential(
          (conv_64): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_64): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_64): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (65): Sequential(
          (shortcut_65): EmptyLayer()
        )
        (66): Sequential(
          (conv_66): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_66): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_66): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (67): Sequential(
          (conv_67): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_67): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_67): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (68): Sequential(
          (shortcut_68): EmptyLayer()
        )
        (69): Sequential(
          (conv_69): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_69): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_69): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (70): Sequential(
          (conv_70): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_70): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_70): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (71): Sequential(
          (shortcut_71): EmptyLayer()
        )
        (72): Sequential(
          (conv_72): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_72): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_72): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (73): Sequential(
          (conv_73): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_73): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_73): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (74): Sequential(
          (shortcut_74): EmptyLayer()
        )
        (75): Sequential(
          (conv_75): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_75): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_75): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (76): Sequential(
          (conv_76): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_76): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_76): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (77): Sequential(
          (conv_77): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_77): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_77): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (78): Sequential(
          (conv_78): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_78): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_78): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (79): Sequential(
          (conv_79): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_79): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_79): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (80): Sequential(
          (conv_80): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_80): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_80): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (81): Sequential(
          (conv_81): Conv2d(1024, 255, kernel_size=(1, 1), stride=(1, 1))
        )
        (82): Sequential(
          (Detection_82): DetectionLayer()
        )
        (83): Sequential(
          (route_83): EmptyLayer()
        )
        (84): Sequential(
          (conv_84): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_84): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_84): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (85): Sequential(
          (upsample_85): Upsample(scale_factor=2.0, mode=nearest)
        )
        (86): Sequential(
          (route_86): EmptyLayer()
        )
        (87): Sequential(
          (conv_87): Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_87): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_87): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (88): Sequential(
          (conv_88): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_88): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_88): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (89): Sequential(
          (conv_89): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_89): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_89): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (90): Sequential(
          (conv_90): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_90): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_90): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (91): Sequential(
          (conv_91): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_91): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_91): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (92): Sequential(
          (conv_92): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_92): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_92): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (93): Sequential(
          (conv_93): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
        )
        (94): Sequential(
          (Detection_94): DetectionLayer()
        )
        (95): Sequential(
          (route_95): EmptyLayer()
        )
        (96): Sequential(
          (conv_96): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_96): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_96): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (97): Sequential(
          (upsample_97): Upsample(scale_factor=2.0, mode=nearest)
        )
        (98): Sequential(
          (route_98): EmptyLayer()
        )
        (99): Sequential(
          (conv_99): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_99): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_99): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (100): Sequential(
          (conv_100): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_100): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_100): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (101): Sequential(
          (conv_101): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_101): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_101): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (102): Sequential(
          (conv_102): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_102): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_102): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (103): Sequential(
          (conv_103): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_103): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_103): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (104): Sequential(
          (conv_104): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_104): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_104): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (105): Sequential(
          (conv_105): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
        )
        (106): Sequential(
          (Detection_106): DetectionLayer()
        )
      )))




```python
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        
    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}   #We cache the outputs for the route layer
        
        write = 0
        for i, module in enumerate(modules):        
            module_type = (module["type"])
            
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
    
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
    
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
    
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
    
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                
    
            elif  module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
    
            elif module_type == 'yolo':        
                anchors = self.module_list[i][0].anchors
                #Get the input dimensions
                inp_dim = int (self.net_info["height"])
        
                #Get the number of classes
                num_classes = int (module["classes"])
        
                #Transform 
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:              #if no collector has been intialised. 
                    detections = x
                    write = 1
        
                else:       
                    detections = torch.cat((detections, x), 1)
        
            outputs[i] = x
        
        return detections
    def load_weights(self, weightfile):
        #Open the weights file
        fp = open(weightfile, "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)



```

# detect


```python
from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
import argparse
import os 
import os.path as osp
import pickle as pkl
import pandas as pd
import random
```


```python
def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    
    return parser.parse_args(args=[])
```


```python
args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()
```


```python
num_classes = 80
classes = load_classes("data/coco.names")
```


```python
#Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32
```

    Loading network.....
    Network successfully loaded



```python
#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()
```


```python
#Set the model in evaluation mode
model.eval()

read_dir = time.time()
```


```python




#Detection phase
try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print ("No file or directory with the name {}".format(images))
    exit()
    
if not os.path.exists(args.det):
    os.makedirs(args.det)

load_batch = time.time()
loaded_ims = [cv2.imread(x) for x in imlist]

im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)


leftover = 0
if (len(im_dim_list) % batch_size):
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover            
    im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                        len(im_batches))]))  for i in range(num_batches)]  

write = 0


if CUDA:
    im_dim_list = im_dim_list.cuda()
    
start_det_loop = time.time()
for i, batch in enumerate(im_batches):
#load the image 
    start = time.time()
    if CUDA:
        batch = batch.cuda()
    with torch.no_grad():
        prediction = model(Variable(batch), CUDA)

    prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)

    end = time.time()

    if type(prediction) == int:

        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue

    prediction[:,0] += i*batch_size    #transform the atribute from index in batch to index in imlist 

    if not write:                      #If we have't initialised output
        output = prediction  
        write = 1
    else:
        output = torch.cat((output,prediction))

    for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()       
try:
    output
except NameError:
    print ("No detections were made")
    exit()

im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

scaling_factor = torch.min(416/im_dim_list,1)[0].view(-1,1)


output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2



output[:,1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
    
    
output_recast = time.time()
class_load = time.time()
colors = pkl.load(open("pallete", "rb"))

draw = time.time()


def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


list(map(lambda x: write(x, loaded_ims), output))

det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))

list(map(cv2.imwrite, det_names, loaded_ims))


end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")


torch.cuda.empty_cache()
```

    d:\resentRes\Pytorch\Pytorch_from_scratch\YOLOv3\imgs\dog.jpg predicted in  0.064 seconds
    Objects Detected:    bicycle truck dog
    ----------------------------------------------------------
    d:\resentRes\Pytorch\Pytorch_from_scratch\YOLOv3\imgs\eagle.jpg predicted in  0.059 seconds
    Objects Detected:    bird
    ----------------------------------------------------------
    d:\resentRes\Pytorch\Pytorch_from_scratch\YOLOv3\imgs\giraffe.jpg predicted in  0.058 seconds
    Objects Detected:    zebra giraffe giraffe
    ----------------------------------------------------------
    d:\resentRes\Pytorch\Pytorch_from_scratch\YOLOv3\imgs\herd_of_horses.jpg predicted in  0.059 seconds
    Objects Detected:    horse horse horse horse
    ----------------------------------------------------------
    d:\resentRes\Pytorch\Pytorch_from_scratch\YOLOv3\imgs\img1.jpg predicted in  0.063 seconds
    Objects Detected:    person dog
    ----------------------------------------------------------
    d:\resentRes\Pytorch\Pytorch_from_scratch\YOLOv3\imgs\img2.jpg predicted in  0.058 seconds
    Objects Detected:    train
    ----------------------------------------------------------
    d:\resentRes\Pytorch\Pytorch_from_scratch\YOLOv3\imgs\img3.jpg predicted in  0.066 seconds
    Objects Detected:    car car car car car car car truck traffic light
    ----------------------------------------------------------
    d:\resentRes\Pytorch\Pytorch_from_scratch\YOLOv3\imgs\img4.jpg predicted in  0.057 seconds
    Objects Detected:    chair chair chair clock
    ----------------------------------------------------------
    d:\resentRes\Pytorch\Pytorch_from_scratch\YOLOv3\imgs\messi.jpg predicted in  0.061 seconds
    Objects Detected:    person person person sports ball
    ----------------------------------------------------------
    d:\resentRes\Pytorch\Pytorch_from_scratch\YOLOv3\imgs\person.jpg predicted in  0.067 seconds
    Objects Detected:    person dog horse
    ----------------------------------------------------------
    d:\resentRes\Pytorch\Pytorch_from_scratch\YOLOv3\imgs\scream.jpg predicted in  0.053 seconds
    Objects Detected:    
    ----------------------------------------------------------
    SUMMARY
    ----------------------------------------------------------
    Task                     : Time Taken (in seconds)
    
    Reading addresses        : 39.411
    Loading batch            : 0.718
    Detection (11 images)    : 0.688
    Output Processing        : 0.000
    Drawing Boxes            : 0.022
    Average time_per_img     : 0.130
    ----------------------------------------------------------


# video detection


```python
from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
import argparse
import os 
import os.path as osp
import pickle as pkl
import pandas as pd
import random
```


```python
args = arg_parse()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()



num_classes = 80
classes = load_classes("data/coco.names")



#Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()


#Set the model in evaluation mode
model.eval()
```

    Loading network.....
    Network successfully loaded

    Darknet(
      (module_list): ModuleList(
        (0): Sequential(
          (conv_0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_0): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_0): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (1): Sequential(
          (conv_1): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (batch_norm_1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_1): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (2): Sequential(
          (conv_2): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_2): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (3): Sequential(
          (conv_3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_3): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (4): Sequential(
          (shortcut_4): EmptyLayer()
        )
        (5): Sequential(
          (conv_5): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (batch_norm_5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_5): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (6): Sequential(
          (conv_6): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_6): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_6): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (7): Sequential(
          (conv_7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_7): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (8): Sequential(
          (shortcut_8): EmptyLayer()
        )
        (9): Sequential(
          (conv_9): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_9): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_9): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (10): Sequential(
          (conv_10): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_10): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_10): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (11): Sequential(
          (shortcut_11): EmptyLayer()
        )
        (12): Sequential(
          (conv_12): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (batch_norm_12): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_12): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (13): Sequential(
          (conv_13): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_13): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_13): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (14): Sequential(
          (conv_14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_14): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_14): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (15): Sequential(
          (shortcut_15): EmptyLayer()
        )
        (16): Sequential(
          (conv_16): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_16): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_16): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (17): Sequential(
          (conv_17): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_17): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_17): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (18): Sequential(
          (shortcut_18): EmptyLayer()
        )
        (19): Sequential(
          (conv_19): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_19): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_19): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (20): Sequential(
          (conv_20): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_20): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_20): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (21): Sequential(
          (shortcut_21): EmptyLayer()
        )
        (22): Sequential(
          (conv_22): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_22): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_22): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (23): Sequential(
          (conv_23): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_23): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_23): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (24): Sequential(
          (shortcut_24): EmptyLayer()
        )
        (25): Sequential(
          (conv_25): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_25): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_25): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (26): Sequential(
          (conv_26): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_26): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_26): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (27): Sequential(
          (shortcut_27): EmptyLayer()
        )
        (28): Sequential(
          (conv_28): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_28): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_28): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (29): Sequential(
          (conv_29): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_29): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_29): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (30): Sequential(
          (shortcut_30): EmptyLayer()
        )
        (31): Sequential(
          (conv_31): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_31): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_31): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (32): Sequential(
          (conv_32): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_32): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_32): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (33): Sequential(
          (shortcut_33): EmptyLayer()
        )
        (34): Sequential(
          (conv_34): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_34): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_34): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (35): Sequential(
          (conv_35): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_35): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_35): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (36): Sequential(
          (shortcut_36): EmptyLayer()
        )
        (37): Sequential(
          (conv_37): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (batch_norm_37): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_37): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (38): Sequential(
          (conv_38): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_38): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_38): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (39): Sequential(
          (conv_39): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_39): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_39): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (40): Sequential(
          (shortcut_40): EmptyLayer()
        )
        (41): Sequential(
          (conv_41): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_41): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_41): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (42): Sequential(
          (conv_42): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_42): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_42): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (43): Sequential(
          (shortcut_43): EmptyLayer()
        )
        (44): Sequential(
          (conv_44): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_44): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_44): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (45): Sequential(
          (conv_45): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_45): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_45): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (46): Sequential(
          (shortcut_46): EmptyLayer()
        )
        (47): Sequential(
          (conv_47): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_47): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_47): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (48): Sequential(
          (conv_48): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_48): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_48): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (49): Sequential(
          (shortcut_49): EmptyLayer()
        )
        (50): Sequential(
          (conv_50): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_50): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_50): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (51): Sequential(
          (conv_51): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_51): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_51): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (52): Sequential(
          (shortcut_52): EmptyLayer()
        )
        (53): Sequential(
          (conv_53): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_53): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_53): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (54): Sequential(
          (conv_54): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_54): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_54): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (55): Sequential(
          (shortcut_55): EmptyLayer()
        )
        (56): Sequential(
          (conv_56): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_56): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_56): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (57): Sequential(
          (conv_57): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_57): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_57): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (58): Sequential(
          (shortcut_58): EmptyLayer()
        )
        (59): Sequential(
          (conv_59): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_59): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_59): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (60): Sequential(
          (conv_60): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_60): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_60): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (61): Sequential(
          (shortcut_61): EmptyLayer()
        )
        (62): Sequential(
          (conv_62): Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (batch_norm_62): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_62): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (63): Sequential(
          (conv_63): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_63): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_63): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (64): Sequential(
          (conv_64): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_64): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_64): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (65): Sequential(
          (shortcut_65): EmptyLayer()
        )
        (66): Sequential(
          (conv_66): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_66): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_66): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (67): Sequential(
          (conv_67): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_67): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_67): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (68): Sequential(
          (shortcut_68): EmptyLayer()
        )
        (69): Sequential(
          (conv_69): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_69): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_69): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (70): Sequential(
          (conv_70): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_70): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_70): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (71): Sequential(
          (shortcut_71): EmptyLayer()
        )
        (72): Sequential(
          (conv_72): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_72): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_72): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (73): Sequential(
          (conv_73): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_73): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_73): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (74): Sequential(
          (shortcut_74): EmptyLayer()
        )
        (75): Sequential(
          (conv_75): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_75): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_75): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (76): Sequential(
          (conv_76): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_76): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_76): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (77): Sequential(
          (conv_77): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_77): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_77): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (78): Sequential(
          (conv_78): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_78): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_78): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (79): Sequential(
          (conv_79): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_79): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_79): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (80): Sequential(
          (conv_80): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_80): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_80): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (81): Sequential(
          (conv_81): Conv2d(1024, 255, kernel_size=(1, 1), stride=(1, 1))
        )
        (82): Sequential(
          (Detection_82): DetectionLayer()
        )
        (83): Sequential(
          (route_83): EmptyLayer()
        )
        (84): Sequential(
          (conv_84): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_84): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_84): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (85): Sequential(
          (upsample_85): Upsample(scale_factor=2.0, mode=nearest)
        )
        (86): Sequential(
          (route_86): EmptyLayer()
        )
        (87): Sequential(
          (conv_87): Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_87): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_87): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (88): Sequential(
          (conv_88): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_88): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_88): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (89): Sequential(
          (conv_89): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_89): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_89): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (90): Sequential(
          (conv_90): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_90): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_90): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (91): Sequential(
          (conv_91): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_91): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_91): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (92): Sequential(
          (conv_92): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_92): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_92): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (93): Sequential(
          (conv_93): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
        )
        (94): Sequential(
          (Detection_94): DetectionLayer()
        )
        (95): Sequential(
          (route_95): EmptyLayer()
        )
        (96): Sequential(
          (conv_96): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_96): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_96): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (97): Sequential(
          (upsample_97): Upsample(scale_factor=2.0, mode=nearest)
        )
        (98): Sequential(
          (route_98): EmptyLayer()
        )
        (99): Sequential(
          (conv_99): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_99): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_99): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (100): Sequential(
          (conv_100): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_100): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_100): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (101): Sequential(
          (conv_101): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_101): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_101): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (102): Sequential(
          (conv_102): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_102): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_102): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (103): Sequential(
          (conv_103): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (batch_norm_103): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_103): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (104): Sequential(
          (conv_104): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_104): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_104): LeakyReLU(negative_slope=0.1, inplace=True)
        )
        (105): Sequential(
          (conv_105): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
        )
        (106): Sequential(
          (Detection_106): DetectionLayer()
        )
      )
    )




```python
#Detection phase

videofile = 'tf3.mp4' #or path to the video file. 

#cap = cv2.VideoCapture(videofile)  

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)#  for webcam

assert cap.isOpened(), 'Cannot capture source'

frames = 0  
start = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:   
        img = prep_image(frame, inp_dim)
#        cv2.imshow("a", frame)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1,2)   
                     
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        
        with torch.no_grad():
            output = model(Variable(img, volatile = True), CUDA)
        output = write_results(output, confidence, num_classes, nms_conf = nms_thesh)


        if type(output) == int:
            frames += 1
            print("FPS of the video is {:5.4f}".format( frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue
        
        
        

        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(416/im_dim,1)[0].view(-1,1)
        
        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
        
        output[:,1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
    
        
        

        classes = load_classes('data/coco.names')
        colors = pkl.load(open("pallete", "rb"))

        list(map(lambda x: write(x, frame), output))
        
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frames += 1
        print(time.time() - start)
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
    else:
        break     

```

### 场景一

![image-20220204183155656](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/image-20220204183155656.png)

### 场景二

![image-20220204183315558](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/image-20220204183315558.png)

显然这里暴漏了初代yolo的缺点 小物体识别较差

### 场景三

![image-20220204183453338](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/image-20220204183453338.png)

### 场景四

![image-20220204183546779](https://fastly.jsdelivr.net/gh/blueflylabor/_ebxeax.github.io@0.0/images/image-20220204183546779.png)

### 场景五 webcam

这里就不展示了哈哈哈

### 场景六 搬运到Android移动端设备

使用Pytorch转译

Android训练就算了


**cv2的webcam或video调用就不赘述了，结案**
