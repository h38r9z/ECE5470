java c
ECE5470   Prelim 3    Fall   2022
1. General (22 pts) 
A. Precision Recall 
The figure to the   right shows the   location of   points   for
plotting a   precision   recall curve.
Numerically evaluate the Average   Precision   (AP)
AP:   1/11((1*5)+(0.57*4)+(0.5*2))=0.752
AUC:   1   * 0.4 + 0.57   *   0.4   +   0.5   *   0.2   =   0.629

B. PR - ROC 
(a)   Under what condition would you   do   an   ROC
analysis   in   preference to a   precision   recall analysis?
When the costs associated with   outcomes   are   different
(b)   Under what condition would you   do   an   PR   analysis   in   preference to an   ROC analysis?
When the   populations of   positives and   negatives are different.
That   is, there are typically   more of one class   than   the   other   in   the   nature.
C. ROC 
A two-class task   has the following outcomes for a test   set.   Carefully sketch the   Receiver Operating Characteristic
(ROC) on   the   right.Index Confidence positive Correct class 1 .97 + 2 .35 + 3 .72 + 4 .72 + 5 .49 + 6 .33 - 7 .25 - 8 .20 - 9 .48 - 10 .12 - 11 .34 - 12 .38 - 13 .82 - 14 .15 - 15 .32 - 
Sensitivity   (TPR)

1 – specificity.      (FPR)
2. Machine Learning (22 pts) 
A. The figure   presents the   learning curve of a   neural   network. As   the   number of epochs   increases from a certain   level, the   test error   is also   increasing. What   is the   name of this   issue   and   how   one can tackle   it? Show you solution   on   the   figure.
(a).    What   is the   name of   the   issue?
Overfitting  
(b). What would you   do to   address   it?
Early stopping    (e.g.   At   750   Epochs)

B. What action would you take for   each   of the   following   cases for   learning curves
(a) The error for the validation set continues   to   decrease   but   the   error   for   the   training   set   begins   to   slowly   increase
_    This should   never   happen. Check the system/program for   errors
(b) The training error changes significantly   (in general   alternating from   positive   to   negative)   between   each   epoch.
Decrease the   Learning   rate
______________________________________________________________________
C.   P   and   N
(a) Give an equation for accuracy   in   terms   ofT,   P,   TP,   FP,   TN,   etc.
(TP + TN)/(TP   + TN   +   FP   +   FN)
(b) Give an equation for sensitivity   in   terms   ofT,   P, TP,   FP,   TN,   etc.
TP/(TP +   FN   )
(c)   Based on the given values, fill the confusion   matrix from   the   given   values.   Label the   rows and columnsAccuracy = 890/1000 Recall = 560/610 Precision = 560/ 620 Total positive values=610 

3. Segmentation and Object recognition (18 pts) 
A. A common component of   a UNet segmentation model is   a transposed   convolution   layer   (a) what is the function of   this   layer?
Up-sampling: to convert a feature image to the next higher resolution 2 x   2   times
(b) how many parameters (weights) does this layer contain (when used   in   a Unet)?
# channels   * 4   +   1   (bias)
(c) what activation function is typically used with   this layer?
Usually, no activation function is used with this layer for U-Net
B. Given a multi-object detection task on 256 x 256   color   images   with   10   object   classes   (a) What is the difference between a Fast RCNN and.   A Faster   RCNN?
A fast RCNN has an algorithm for region selection   whereas   a   Faster   RCNN   has   a   trained   Region   Proposal
Network (RPN)    
(b)    How many (multiclass) outputs does a YOLO   Segmentation model have?
N x N where N is a fraction   of   256   
(c) For a Region-based Fully Convolutional Network (R-FCN)   How many outputs are trained for   each   class?
9  
Explain why multiple outputs are trained
The 9 outputs correspond to nine regions (3 x   3) that   cover the   object. All   the   outputs   for   each   “zone”   should   respond   to   indicate   the代 写ECE5470 Prelim 3 Fall 2022Matlab
代做程序编程语言   presence   of   the   target   object.
C. A   UNet or   FCNN   used to   locate the   pixels of small   objects   has two   issues   with   respect   to   the   loss   function.   What are these   issues and   how would you   address them?
1. The   metric   used for testing   is   DICE score   no accuracy   (pixels correct).   Solution   used   loss   based   on   DICE   score
2. There are   many   more   positive than   negative outcomes   in the training set;   i.e.,   unbalanced   training   set.
Use a weighted   loss function that   has a   higher   loss   magnitude for   positive   errors   compared to   negative   errors.
4. CNN Model (20 pts.) 
# Convolutional neural network (two convolutional layers) 
class ConvNet (nn.Module): 
def   init  (self, num_classes=9): 
super (ConvNet, self).  init  () 
self.layer1 = nn.Sequential ( 
nn.Conv2d (1, 16, kernel_size=3, stride=1, padding=1), 
nn.BatchNorm2d (16), 
nn.ReLU (), 
nn.MaxPool2d (kernel_size=2, stride=2)) 
self.layer2 = nn.Sequential ( 
nn.Conv2d (16, 16, kernel_size=5, stride=1, padding=2), 
nn.BatchNorm2d (32), 
nn.ReLU (), 
nn.MaxPool2d (kernel_size=2, stride=2)) 
self.layer3 = nn.Sequential ( 
nn.Conv2d (16, 32, kernel_size=3, stride=1, padding=1), 
nn.BatchNorm2d (32), 
nn.ReLU (), 
nn.MaxPool2d (kernel_size=2, stride=2)) 
self.fc = nn.Linear (7*7*32, num_classes) 
def forward (self, x): 
out = self.layer1 (x) 
out = self.layer2 (out) 
out = self.layer3 (out) 
out = out.reshape (out.size (0), -1) 
out = self.fc (out) 
return out 
The model is designed to classify images of   size   128 x   128   into   9   different   classes   (a) How many weights/parameters are associated with layer1:
160  
(b) How many weights/parameters are associated with layer2:
6416   
(c) How many weights/parameters are associated with layer3:
4640   
(d) How many weights/parameters are associated with layer fc:
73737  
For a specific problem a designer decides to preprocess the image with   the   Fourier   transform.	
(e) What modification, if   any, would need to be made to the model to function with the transformed image?
2 input channels   instead   of 1   
(f) By using the Fourier transform. which of   the following might the designer expect the performance to be   less sensitive to:
(a) translation (b) rotation, (c) scale   (d)   aspect ratio)
______a____________________________________________
5 .Misc (22 pts) 
A. CT: Computerized Tomography (CT) scanners are   used for   a   number   of   applications   including   medical diagnosis and baggage inspection. To design   a   deep   learning   image   analysis   model for   images   created   by   CT   scanners requires resolving a   number   of   issues.
List three main ways in which CT   images differ from   traditional   camera   images.
1.   Calibrated   images  
2.  3D and   large  
3.  Grayscale   
Indicate how you what you would do differently for CT   images with   respect ot   the   following
1.   Image   Preprocessing   No standardization  
2.   Model   Design   3D   instead   of 2D   
3. Training   Smaller batch size/   different   loss   function  
B. Training
1. Which of the following techniques can   be   used to   reduce   model   overfitting?
(a)   Data augmentation,   (b)   Dropout,   (c)   Batch   Normalization   (d)   Using Adam   instead of SGD
A,B,C   
C.   Models
(a) What   is the   main feature of   inception   net   models for   performance   improvement?
Leveraging different filter sizes   at   once   
(a) What   is the   main feature of   resnet   models for   performance   improvement?
Skip connections to combat vanishing gradient   
(c) An   inception   module   is   shown on the   right.
Clearly   label the function of
each   box   including the function
“size”   parameter.
[do   not   include
number of   channels]


         
加QQ：99515681  WX：codinghelp  Email: 99515681@qq.com
