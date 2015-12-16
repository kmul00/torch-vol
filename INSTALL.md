Each module can be associated with any of the following:

* A ***`lua`*** file which is the main part of the module. 
* The *`lua`* script can call an underlying ***`C`*** implementation which runs on CPU (can be found under the [generic](https://github.com/kmul00/torch-vol/tree/master/generic) folder).
* The *`lua`* script can also call a ***`cuda`*** implementation to enable GPU execution (can be found under the [cuda](https://github.com/kmul00/torch-vol/tree/master/cuda) folder).

Below each module, in the [README](https://github.com/kmul00/torch-vol/blob/master/README.md), I have mentioned the associated files (`lua` or `c` or `cuda`).

We will go through the ways to install each module.

Let's say you want to install the **VolumetricConvolution** module.

### Installing `lua` modules

1. `git clone` the [torch nn](https://github.com/torch/nn) repository to your `torch_home/extra` directory
2. `cd nn`
3. Download a copy of `VolumetricConvolution.lua` (from [here](https://github.com/kmul00/torch-vol)) to the `torch_home/extra/nn` directory
4. Edit `init.lua` (in `torch_home/extra/nn`) with your favorite editor
5. Make an entry like `include('VolumetricConvolution.lua')`
6. `save` and `exit` editor
7. Run `luarocks make rocks/nn-scm-1.rockspec`

### Installing `C` modules

1. Assuming you have already performed steps **1-7** as mentioned in **Installing `lua` modules**
2. `cd generic`
3. Download a copy of `VolumetricConvolutionMM.c` (from [here](https://github.com/kmul00/torch-vol/tree/master/generic)) to the `torch_home/extra/nn/generic` directory
4. `cd ..`
5. Edit `init.c` (in `torch_home/extra/nn`) with your favorite editor
6. Add the following entries under their respective sections (easily understandable once you have a look at the file)
  1. `#include "generic/VolumetricConvolutionMM.c"`
  2. `#include "THGenerateFloatTypes.h"`
  3. `nn_FloatVolumetricConvolutionMM_init(L);` (notice the trailing *`_init`*)
  4. `nn_DoubleVolumetricConvolutionMM_init(L);` (notice the trailing *`_init`*)
7. `save` and `exit` editor
8. Run `luarocks make rocks/nn-scm-1.rockspec`

### Installing `cuda` modules

1. Assuming you have already performed steps **1-8** as mentioned in **Installing `lua` modules**
2. `git clone` the [torch cunn](https://github.com/torch/cunn) repository to your `torch_home/extra` directory
3. `cd cunn`
4. Download a copy of `VolumetricConvolution.cu`  (from [here](https://github.com/kmul00/torch-vol/tree/master/cuda)) to the `torch_home/extra/cunn` directory
5. Edit `init.cu` (in `torch_home/extra/cunn`) with your favorite editor
6. Make an entry like `cunn_VolumetricConvolution_init(L);` (notice the trailing *`_init`*)
7. `save` and `exit` editor
8. Edit `utils.h` (in `torch_home/extra/cunn`) with your favorite editor
9. Make an entry like `void cunn_VolumetricConvolution_init(lua_State *L);` (notice the trailing *`_init`*)
10. `save` and `exit` editor
11. Run `luarocks make rocks/cunn-scm-1.rockspec`


Once you have executed the above steps without any error, you are good to go and can use the installed module like any other Torch module.
