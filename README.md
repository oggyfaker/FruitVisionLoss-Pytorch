# Fruit Vision Loss
My implementation of label-smooth, amsoftmax, focal-loss, dual-focal-loss, triplet-loss, giou-loss, affinity-loss, pc_softmax_cross_entropy, ohem-loss(softmax based on line hard mining loss), large-margin-softmax(bmvc2019), lovasz-softmax-loss, and dice-loss(both generalized soft dice loss and batch soft dice loss). Maybe this is useful in my future work.


Also tried to implement swish, hard-swish(hswish) and mish activation functions.

Additionally, cuda based one-hot function is added (support label smooth).

Newly add an "Exponential Moving Average(EMA)" operator.

Add convolution ops, such as coord-conv2d, and dynamic-conv2d(dy-conv2d).

Some operators are implemented with pytorch cuda extension, so you need to compile it first: 
```
    $ python setup.py install
```

After installing, now you can pick up what you need and use the losses or ops like one of thes: 


```python
from FruitVisionLoss-Pytorch import SwishV1, SwishV2, SwishV3
from FruitVisionLoss-Pytorch import HSwishV1, HSwishV2, HSwishV3
from FruitVisionLoss-Pytorch import MishV1, MishV2, MishV3
from FruitVisionLoss-Pytorch import convert_to_one_hot, convert_to_one_hot_cu, OnehotEncoder
from FruitVisionLoss-Pytorch import EMA

from FruitVisionLoss-Pytorch import TripletLoss
from FruitVisionLoss-Pytorch import SoftDiceLossV1, SoftDiceLossV2, SoftDiceLossV3
from FruitVisionLoss-Pytorch import PCSoftmaxCrossEntropyV1, PCSoftmaxCrossEntropyV2
from FruitVisionLoss-Pytorch import LargeMarginSoftmaxV1, LargeMarginSoftmaxV2, LargeMarginSoftmaxV3
from FruitVisionLoss-Pytorch import LabelSmoothSoftmaxCEV1, LabelSmoothSoftmaxCEV2, LabelSmoothSoftmaxCEV3
from FruitVisionLoss-Pytorch import generalized_iou_loss
from FruitVisionLoss-Pytorch import FocalLossV1, FocalLossV2, FocalLossV3
from FruitVisionLoss-Pytorch import Dual_Focal_loss
from FruitVisionLoss-Pytorch import GeneralizedSoftDiceLoss, BatchSoftDiceLoss
from FruitVisionLoss-Pytorch import AMSoftmax
from FruitVisionLoss-Pytorch import AffinityFieldLoss
from FruitVisionLoss-Pytorch import OhemCELoss, OhemLargeMarginLoss
from FruitVisionLoss-Pytorch import LovaszSoftmax
from FruitVisionLoss-Pytorch import TaylorCrossEntropyLoss

from FruitVisionLoss-Pytorch import TaylorSoftmax

from FruitVisionLoss-Pytorch import CoordConv2d, DY_Conv2d
```
Note that some losses or ops have 3 versions, like `LabelSmoothSoftmaxCEV1`, `LabelSmoothSoftmaxCEV2`, `LabelSmoothSoftmaxCEV3`, here `V1` means the implementation with pure pytorch ops and use `torch.autograd` for backward computation, `V2` means implementation with pure pytorch ops but use self-derived formula for backward computation, and `V3` means implementation with cuda extension. Generally speaking, the `V3` ops are faster and more memory efficient, since I have tried to squeeze everything in one cuda kernel function, which in most cases brings less overhead than a combination of pytorch ops.


For those who happen to find this repo, if you see errors in my code, feel free to open an issue to correct me.
