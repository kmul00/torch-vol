#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricUpSamplingNearest.c"
#else

static int nn_(VolumetricUpSamplingNearest_updateOutput)(lua_State *L)
{
  // get all params
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int scale_factor_t = luaT_getfieldcheckint(L, 1, "scale_factor_t");
  int scale_factor_xy = luaT_getfieldcheckint(L, 1, "scale_factor_xy");
  int dT = scale_factor_t;
  int dW = scale_factor_xy;
  int dH = scale_factor_xy;
  int tDim = input->nDimension-3;
  int xDim = input->nDimension-2;
  int yDim = input->nDimension-1;
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  // dims
  int idim = input->nDimension;  
  int osz0 = output->size[0];
  int osz1 = output->size[1];
  int osz2 = output->size[2];
  int osz3 = output->size[3];
  int osz4 = 1;
  if (idim > 4) {
    osz4 = output->size[4];
  }

  // get strides
  long *is = input->stride;
  long *os = output->stride;

  // get raw pointers
  real *pin = THTensor_(data)(input);
  real *pout = THTensor_(data)(output);

  // perform the upsampling
  int i0, i1, i2, i3, i4, isrc, idst;
  int iout[5];  // Output indices
  int iin[5];  // Input indices

  for (i0 = 0; i0 < osz0; i0++) {
    iout[0] = i0;
    iin[0] = i0;
    for (i1 = 0; i1 < osz1; i1++) {
      iout[1] = i1;
      iin[1] = i1;
      for (i2 = 0; i2 < osz2; i2++) {
        iout[2] = i2;
        iin[2] = i2;
        for (i3 = 0; i3 < osz3; i3++) {
          iout[3] = i3;
          iin[3] = i3;
          for (i4 = 0; i4 < osz4; i4++) {
            iout[4] = i4;
            iin[4] = i4;


            // set the indices for the upsampled dimensions
            iin[tDim] = iout[tDim] / dT;
            iin[xDim] = iout[xDim] / dW;
            iin[yDim] = iout[yDim] / dH;

            idst = i0*os[0] + i1*os[1] + i2*os[2] + i3*os[3];
            isrc = iin[0]*is[0] + iin[1]*is[1] + iin[2]*is[2] + iin[3]*is[3];
            if (idim > 4) {
              idst += i4*os[4];
              isrc += iin[4]*is[4];
            }
            pout[idst] = pin[isrc];
          }
        }
      }
    }
  }
  return 1;
}

static int nn_(VolumetricUpSamplingNearest_updateGradInput)(lua_State *L)
{
  // get all params
  //THTensor *input = luaT_checkudata(L,2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L,3, torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L,1, "gradInput", torch_Tensor);

  int scale_factor_t = luaT_getfieldcheckint(L, 1, "scale_factor_t");
  int scale_factor_xy = luaT_getfieldcheckint(L, 1, "scale_factor_xy");
  int dT = scale_factor_t;
  int dW = scale_factor_xy;
  int dH = scale_factor_xy;
  int tDim = gradInput->nDimension-3;
  int xDim = gradInput->nDimension-2;
  int yDim = gradInput->nDimension-1;
  
  // dims
  int idim = gradInput->nDimension;  
  int isz0 = gradInput->size[0];
  int isz1 = gradInput->size[1];
  int isz2 = gradInput->size[2];
  int isz3 = gradInput->size[3];
  int isz4 = 1;
  if (idim > 4) {
    isz4 = gradInput->size[4];
  }

  // get strides
  long *is = gradInput->stride;
  long *os = gradOutput->stride;

  // get raw pointers
  real *pin = THTensor_(data)(gradInput);
  real *pout = THTensor_(data)(gradOutput);

  // perform the upsampling
  int i0, i1, i2, i3, i4, isrc, idst, x, y, t;
  int iout[5];  // Output indices
  int iin[5];  // Input indices

  THTensor_(zero)(gradInput);

  for (i0 = 0; i0 < isz0; i0++) {
    iin[0] = i0;
    iout[0] = i0;
    for (i1 = 0; i1 < isz1; i1++) {
      iin[1] = i1;
      iout[1] = i1;
      for (i2 = 0; i2 < isz2; i2++) {
        iin[2] = i2;
        iout[2] = i2;
        for (i3 = 0; i3 < isz3; i3++) {
          iin[3] = i3;
          iout[3] = i3;
          for (i4 = 0; i4 < isz4; i4++) {
            iin[4] = i4;
            iout[4] = i4;

            idst = i0*is[0] + i1*is[1] + i2*is[2] + i3*is[3];
            if (idim > 4) {
              idst += i4*is[4];
            }

            // Now accumulate the gradients from gradOutput
            for (t = 0; t < dT; t++) {
              for (y = 0; y < dH; y++) {
                for (x = 0; x < dW; x++) {
                  iout[tDim] = dT * iin[tDim] + t;
                  iout[xDim] = dW * iin[xDim] + x;
                  iout[yDim] = dH * iin[yDim] + y;
                  isrc = iout[0]*os[0] + iout[1]*os[1] + iout[2]*os[2] + iout[3]*os[3];
                  if (idim > 4) {
                    isrc += iout[4]*os[4];
                  }
                  pin[idst] += pout[isrc];
                }
              }
            }
          }
        }
      }
    }
  }
  return 1;
}

static const struct luaL_Reg nn_(VolumetricUpSamplingNearest__) [] = {
  {"VolumetricUpSamplingNearest_updateOutput", nn_(VolumetricUpSamplingNearest_updateOutput)},
  {"VolumetricUpSamplingNearest_updateGradInput", nn_(VolumetricUpSamplingNearest_updateGradInput)},
  {NULL, NULL}
};

static void nn_(VolumetricUpSamplingNearest_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(VolumetricUpSamplingNearest__), "nn");
  lua_pop(L,1);
}

#endif
