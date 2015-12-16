# Volumetric extensions to Torch's modules

In this repository I will be maintaining a list of modules that I had to come up with to ease my work. While starting out to work with ***Volumetric*** (**4D** or *batchmode* **5D**) data I found a lot of implementations missing, even though their ***Spatial*** versions existed. This repository is the collection of the codes that I wrote then. 

*Hopefully it will be of help to someone, someday*.

## Note
***
Most of these are simple extensions of their ***Spatial*** counter-parts. So there's a high chance that even though it wasn't there when I wrote it, it may have been incorporated in the official library now. So it is highly recommended that you cross-check with [Torch's main repository](https://github.com/torch/nn) to see if it already exists. I will notify the same, below each module, if I happen to come across one.

## List of Modules
***

###VolumetricConvolution
___
Performs `Volumetric Convolution` that supports ***padding***

    nn.VolumetricConvolution(nInputPlane, nOutputPlane, kT, kW, kH, [dT], [dW], [dH], [padT], [padW], [padH])

This module is already *[merged](https://github.com/torch/nn/pull/481)* into the main **Torch** repository.

###VolumetricBatchNormalization
___
Performs `Batch Normalization` over ***5D*** (batch mode) data

    nn.VolumetricBatchNormalization(N [,eps] [, momentum] [,affine])

where `N = Number of input features`. Details regarding the other parameters could be found over 
[here](https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialBatchNormalization)

###VolumetricUpSamplingNearest
___
Performs `3D upsampling` on input ***videos*** containing any number of `input planes`.

    nn.VolumetricUpSamplingNearest(scale_t, scale_xy)

where `scale_t = upsample ratio along time domain. scale_xy = upsample ratio along height, width dimension. Must be positive integers.` I have personally used this to perform ***unpooling***. Use case in ***Spatial*** domain could be found in the [dc-ign](http://arxiv.org/pdf/1503.03167v4.pdf) paper.

## Ending Note
***

I will keep on adding stuffs as and when required. If you need anything in particular, you are most welcome to ask about it. Also feel free to give suggestions, comments. It will be much appreciated.

And finally I would like to thank all the wonderful [Torch's contributers](https://github.com/torch/nn/graphs/contributors) for actively maintaining such a wonderful and easy-to-use library. Really appreciate your efforts and hard work.
