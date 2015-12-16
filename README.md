# Volumetric extensions to Torch's modules

In this repository I will be maintaining a list of modules that I had to come up with to ease my work. While starting out to work
with ***Volumetric*** (**4D** or **5D** (*batchmode*) data) I found a lot of implementations missing, even though their ***Spatial***
versions existed. This repository is the collection of the codes that I wrote then. 

Hopefully it will be of help to someone, someday.

## Note
***
Most of these are simple extensions of their ***Spatial*** counter-parts. So there's a high chance that even though it wasn't
there when I wrote it, it may have been incorportaed in the official library now. So it is highly recommended that you cross-check
with [Torch's main repository](https://github.com/torch/nn) to see if it already exists.

## List of Modules
***

###Volumetric Convolution
___
Performs `VolumetricConvolution` that supports *padding*

    nn.VolumetricConvolution(nInputPlane, nOutputPlane, kT, kW, kH, [dT], [dW], [dH], [padT], [padW], [padH])

This module is already *merged* into the main **Torch** repository. Details could be found [here](https://github.com/torch/nn/pull/481)

###Volumetric Batch Normalization
___
Performs `Batch Normalization` over **5D** (batch mode) data

    nn.VolumetricBatchNormalization(N [,eps] [, momentum] [,affine])

`N = Number of input features`. Details regarding the other parameters could be found over 
[here](https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialBatchNormalization)

###Volumetric Up Sampling Nearest
___
Performs `VolumetricConvolution` that supports *padding*

    nn.VolumetricConvolution(nInputPlane, nOutputPlane, kT, kW, kH, [dT], [dW], [dH], [padT], [padW], [padH])

This module is already *merged* into the main **Torch** repository. Details could be found [here](https://github.com/torch/nn/pull/481)


