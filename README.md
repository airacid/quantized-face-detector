# Quantized [DSFD](https://arxiv.org/abs/1810.10220?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+%28ExcitingAds%21+cs+updates+on+arXiv.org%29)


## Introduction

Weight and Activation quantization based on Dual shot face detector (DSFD) implmented with tensorflow. Trained with quantization aware training (QAT) method.


## Requirment

+ tensorflow-mkl1.14 

+ opencv

+ python 3.6

## Benefits of using Binarization and Quantization in Neural Networks
- Uses lower bit-depth compared to the standard network
- Implementing the network in lower bit-depth allows us to deploy it in the embedded devices or microprocessor with the fixed-point processor
- Gains four times of reduction in RAM just by switching from 32-bits to 8-bits
- Lighter deployment models use less storage space, are easier to transfer over lower bandwidths, and are easier to update

## Quantization-Aware Training on Dual-Shot Face Detector
- Pretrained DSFD is quantized from 32-bit to 8-bit precision, then fine-tuned to boost the detection performance
- The size of the network reduced down 60% compared to the original network, while the detection performance degraded by 4.88% only
- The quantized network is more efficient optimization method compared to AI pruning

## Results
### The size and the number of parameters of the original network and the pruned network
|                            |          Size     |         Type    |     The number of parameters |
|----------------------------|:-----------------:|:---------------:|:----------------------------:|
|     Standard   Network     |     170.7   MB    |       Float32   |           42,748,434         |
|     Pruned   Network       |     134.6   MB    |       Float32   |           33,680,043         |
|     Quantized   Network    |     103.9   MB    |    Int8+Float32 |           42,571,021         |
### Average precision comparison on WIDER FACE validation dataset
|                            |     AP   on easy set    |     AP   on medium set    |     AP   on hard set    |
|----------------------------|:-----------------------:|:-------------------------:|:-----------------------:|
|     Standard   Network     |          0.9431         |           0.9320          |          0.8569         |
|     Pruned   network       |          0.9129         |           0.8966          |          0.7763         |
|     Quantized   Network    |          0.9110         |           0.9011          |          0.8151         |

## Visualization
Download the accelerated face detector...

`python vis.py --input IMAGE_PATH`


### References
[DSFD: Dual Shot Face Detector](https://arxiv.org/abs/1810.10220?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+%28ExcitingAds%21+cs+updates+on+arXiv.org%29)
[DSFD-tensorflow implementation](https://github.com/610265158/DSFD-tensorflow)
