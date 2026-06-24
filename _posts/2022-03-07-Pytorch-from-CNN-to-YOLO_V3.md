---
layout: post
title: "PyTorch Deep Dive: From CNN Foundations to YOLOv3 Implementation"
date: 2022-03-07
last_modified_at: 2026-06-23
categories: archive
tags: [PyTorch, CNN, YOLO, Object Detection]
lang: en
---

# Part 1. Foundations of Convolutional Neural Networks (CNN)

## 1. The Dimensionality Explosion in Multi-Layer Perceptrons (MLP)

Consider a binary classification task: separating images of cats and dogs.
Using a modern camera to capture standard images yields a resolution of roughly **12 Megapixels**.

* **Input Size:** A single 3-channel RGB image contains $12\text{M} \times 3 = 36\text{M}$ numerical features.
* **MLP Scaling:** If we connect this input to a single hidden layer with just 100 hidden units, the weight matrix alone would require:
  $$36\text{M} \times 100 = 3.6 \times 10^{9} \text{ parameters } (\approx 14\text{GB of VRAM})$$

This model architecture boasts more parameters than the global census of cats and dogs combined. To fix this over-parameterization, we must incorporate two spatial inductive biases: **Translation Invariance** and **Locality**.

---

## 2. Re-examining the Fully Connected Layer

Let the input $\mathbf{X}$ and the hidden representations $\mathbf{H}$ be structured as 2D spatial matrices instead of flattened vectors. The weights link every input coordinate $(k, l)$ to every output coordinate $(i, j)$, requiring a 4D tensor $\mathbf{W}$:

$$h_{i,j} = \sum_{k,l} W_{i,j,k,l} x_{k,l}$$

By changing indexes with a spatial offset $(a, b) = (k - i, l - j)$, we can rewrite the 4D weight tensor as $\mathbf{V}$ where $V_{i,j,a,b} = W_{i,j,i+a,j+b}$:

$$h_{i,j} = \sum_{a,b} V_{i,j,a,b} x_{i+a,j+b}$$

### Principle #1: Translation Invariance
An object's identity does not change when it shifts across an image. Therefore, the structural response $\mathbf{V}$ should not depend on the absolute spatial output coordinates $(i, j)$. This forces $V_{i,j,a,b} = V_{a,b}$, converting the operation into a standard **Cross-Correlation**:

$$h_{i,j} = \sum_{a,b} V_{a,b} x_{i+a,j+b}$$

### Principle #2: Locality
To evaluate local features at $(i, j)$, we should not process pixels located excessively far away. Thus, for any spatial offset beyond a specific radius $|a| > \Delta$ or $|b| > \Delta$, we set $V_{a,b} = 0$. 

Introducing a bias term $u$, the formulation transforms into a localized spatial convolution:

$$H_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} V_{a, b} X_{i+a, j+b}$$



Using convolutional constraints, parameter counts plunge from billions to hundreds. However, this relies heavily on the inductive bias matching reality; if an application breaks translation invariance, CNNs may struggle to fit the training data.

---

## 3. Spatial Aggregation: Max Pooling

Max Pooling abstracts spatial representation by extracting the maximum activation within a sliding kernel window.



While Max Pooling provides slight translation and deformation tolerance, standard CNN architectures still cannot naturally generalize to massively scaled or rotated targets. To mitigate this structural limitation, aggressive **Data Augmentation** (random scaling, rotation, cropping) remains mandatory during training pipelines.

---

# Part 2. Object Detection with YOLOv1 and YOLOv3

## 1. The Bounding Box Paradigm

YOLO splits the input image into an $S \times S$ grid. If an object's absolute ground-truth center (midpoint) falls inside a specific grid cell, that cell is assigned responsibility for detecting that object.



* **Ground-Truth Target Structure:** For each grid cell, the ground-truth vector is defined as:
  $$\text{label}_{\text{cell}} = [C_1, C_2, \dots, C_{20}, P_c, X, Y, W, H]$$
  Where $P_c \in \{0, 1\}$ indicates object presence, and $[X, Y, W, H]$ defines the relative bounding box.
* **Coordinate Mapping:** $X, Y \in [0, 1]$ are strictly bounded relative offsets within the specific grid cell boundary. $W, H$ represent structural multipliers normalized relative to the full image dimensions (and can thus exceed $1.0$).

---

## 2. Model Implementations via PyTorch

Below is the structured implementation for parsing configurations, manipulating prediction tensors, executing Non-Maximum Suppression (NMS), and building the Darknet backbone network.

### 2.1 Bounding Box Math & Tensor Utilities

```python
import torch
import torch.nn as nn
import numpy as np
import cv2

def unique(tensor):
    """Extracts unique class elements from a 1D tensor safely."""
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def bbox_iou(box1, box2):
    """Calculates intersection-over-union (IoU) scores between two batches of boxes."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    return iou

```

### 2.2 Prediction Vector Transformation & Post-Processing

```python
def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    """Transforms raw network output feature maps into predictable box coordinates."""
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # Map center coordinates and objectness scores through Sigmoid activation
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    # Calculate absolute grid location offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    prediction[:,:,:2] += x_y_offset

    # Apply anchor scale dimensions via log-space transforms
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4]) * anchors
    
    # Class probability activations
    prediction[:,:,5:5+num_classes] = torch.sigmoid((prediction[:,:,5:5+num_classes]))
    prediction[:,:,:4] *= stride
    
    return prediction

def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    """Filters target predictions via thresholding and Non-Maximum Suppression (NMS)."""
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask
    
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = prediction.size(0)
    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]
        
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        image_pred = torch.cat((image_pred[:, :5], max_conf, max_conf_score), 1)
        
        non_zero_ind = torch.nonzero(image_pred[:, 4])
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue
            
        if image_pred_.shape[0] == 0:
            continue       
            
        img_classes = unique(image_pred_[:, -1])
        
        for cls in img_classes:
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)
            
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)
            
            for i in range(idx):
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except (ValueError, IndexError):
                    break
            
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
            
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)
                
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
    try:
        return output
    except:
        return 0

```

### 2.3 Image Preprocessing Pipeline

```python
def letterbox_image(img, inp_dim):
    """Resizes an image using padding while preserving original aspect ratio."""
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h, (w-new_w)//2:(w-new_w)//2 + new_w, :] = resized_image
    return canvas

def prep_image(img, inp_dim):
    """Prepares standard OpenCV image matrices into normalized float Torch Tensors."""
    img = letterbox_image(img, (inp_dim, inp_dim))
    img = img[:,:,::-1].transpose((2,0,1)).copy() # BGR to RGB, then HWC to CHW
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

```

### 2.4 Parsing the Darknet Configuration File (.cfg)

```python
def parse_cfg(cfgfile):
    """Parses structural text blocks from Darknet architecture network files."""
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0 and x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]
    
    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()     
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks

```

---

## 3. Assembling the Darknet Modular Infrastructure

In PyTorch, a custom layer must explicitly detail structural state maps within its sub-classed `forward` step. However, building boilerplate modules for structural manipulation layers like `Route` or `Shortcut` creates unnecessary abstraction overhead.

Instead, we place a dummy module (`EmptyLayer`) into our generated sequential chain. The conditional tensor slicing, channel concatenation (`torch.cat`), and residual additions are then handled directly inside the main model definition's `forward` method.

```python
class EmptyLayer(nn.Module):
    """A placeholder layer used for structural routing logic (Route & Shortcut)."""
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    """A placeholder detection block initialized with specific scaled anchor tensors."""
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors       

def create_modules(blocks):
    """Generates an evaluation layer sequence from raw block configurations."""
    net_info = blocks[0]     
    module_list = nn.ModuleList()
    prev_filters = 3          # Starts at 3 for standard RGB processing
    output_filters = []
    
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        
        if (x["type"] == "convolutional"):
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
        
            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            pad = (kernel_size - 1) // 2 if padding else 0
        
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module(f"conv_{index}", conv)
        
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f"batch_norm_{index}", bn)
        
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module(f"leaky_{index}", activn)
        
        elif (x["type"] == "upsample"):
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module(f"upsample_{index}", upsample)
                
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            start = int(x["layers"][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0
                
            if start > 0: start = start - index
            if end > 0: end = end - index
            
            route = EmptyLayer()
            module.add_module(f"route_{index}", route)
            
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
    
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module(f"shortcut_{index}", shortcut)
            
        elif x["type"] == "yolo":
            mask = [int(mask_idx) for mask_idx in x["mask"].split(",")]
            anchors = [int(a) for a in x["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
    
            detection = DetectionLayer(anchors)
            module.add_module(f"Detection_{index}", detection)
                             
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        
    return (net_info, module_list)

```

### 3.1 Overriding Forward Computations (The Darknet Engine)

The main model orchestration loop processes the routing paths and handles the multi-scale outputs of YOLOv3.

```python
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        
    def forward(self, x, CUDA=True):
        modules = self.blocks[1:]
        outputs = {}  # Caches layer feature maps for Route/Shortcut connections
        write = 0     # Accumulator flag for multi-scale YOLO detections
        
        for i, module in enumerate(self.module_list):        
            module_type = (modules[i]["type"])
            
            if module_type == "convolutional" or module_type == "upsample":
                x = module(x)
                
            elif module_type == "route":
                layers = [int(lyr) for lyr in modules[i]["layers"]]
                if layers[0] > 0: layers[0] = layers[0] - i
                
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] > 0: layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                    
            elif module_type == "shortcut":
                from_layer = int(modules[i]["from"])
                x = outputs[i - 1] + outputs[i + from_layer]
                
            elif module_type == "yolo":
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info["width"])
                num_classes = int(modules[i]["classes"])
                
                # Transform feature map into prediction vectors
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                
                if not write:
                    output = x
                    write = 1
                else:
                    output = torch.cat((output, x), 1)
                    
            outputs[i] = x
            
        return output  # Returns concatenated multi-scale box detection predictions

```

---

## 4. Verification

```python
if __name__ == "__main__":
    # Initialize network model structure
    model = Darknet("cfg/yolov3.cfg")
    print("Darknet modules successfully initiated.")
    
    # Target shape input expectation check
    # Dynamic dimension: Batch Size=1, Channels=3, Height=416, Width=416
    mock_batch = torch.randn(1, 3, 416, 416)
    
    # Disable VRAM acceleration if local device testing environment lacks CUDA
    device_has_cuda = torch.cuda.is_available()
    if device_has_cuda:
        model = model.cuda()
        mock_batch = mock_batch.cuda()
        
    with torch.no_grad():
        predictions = model(mock_batch, CUDA=device_has_cuda)
        
    print(f"Prediction output evaluation tensor shape: {predictions.shape}")
    # Expected output: torch.Size([1, 10647, 85]) 
    # (10647 total anchor box evaluation points for a 416x416 resolution input)

```

```

