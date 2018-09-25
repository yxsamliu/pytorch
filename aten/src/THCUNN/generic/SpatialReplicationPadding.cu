#include "hip/hip_runtime.h"
#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialReplicationPadding.cu"
#else

void THNN_(SpatialReplicationPadding_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int padL, int padR,
           int padT, int padB) {
  THArgCheck(THCTensor_canUse32BitIndexMath(state, input), 2,
             "input tensor must fit into 32-bit index math");

  int planeDim = 0;
  int dimh = 1;
  int dimw = 2;
  int numBatch = 1;

  int numInputDims = THCTensor_(nDimensionLegacyNoScalars)(state, input);
  THCUNN_argCheck(state, !input->is_empty() && (numInputDims == 3 || numInputDims == 4), 2, input,
                  "non-empty 3D or 4D (batch mode) tensor expected for input, but got: %s")

  if (numInputDims == 4) {
    numBatch = THCTensor_(size)(state, input, 0);
    planeDim++;
    dimh++;
    dimw++;
  }

  int numPlanes = THCTensor_(size)(state, input, planeDim);
  int inputH = THCTensor_(size)(state, input, dimh);
  int inputW = THCTensor_(size)(state, input, dimw);
  int outputH = inputH + padT + padB;
  int outputW  = inputW + padL + padR;

  THArgCheck(outputW >= 1 || outputH >= 1 , 2,
             "input (H: %d, W: %d)is too small."
             " Calculated output H: %d W: %d",
             inputH, inputW, outputH, outputW);

  THCDeviceTensor<scalar_t, 4> devInput;
  THCDeviceTensor<scalar_t, 4> devOutput;

  if (numInputDims == 3) {
    THCTensor_(resize3d)(state, output, numPlanes, outputH, outputW);

    devInput = toDeviceTensor<scalar_t, 3>(state, input).upcastOuter<4>();
    devOutput = toDeviceTensor<scalar_t, 3>(state, output).upcastOuter<4>();
  } else {
    THCTensor_(resize4d)(state, output, numBatch, numPlanes, outputH, outputW);

    devInput = toDeviceTensor<scalar_t, 4>(state, input);
    devOutput = toDeviceTensor<scalar_t, 4>(state, output);
  }

  int outputPlaneSize = devOutput.getSize(2) * devOutput.getSize(3);
  dim3 gridSize(THCCeilDiv(outputPlaneSize, 256),
            devOutput.getSize(1),
            devOutput.getSize(0));
  dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

 hipLaunchKernelGGL( SpatialReplicationPadding_updateOutput<scalar_t>, dim3(gridSize), dim3(blockSize), 0, THCState_getCurrentStream(state), 
    devInput, devOutput, static_cast<int>(padT), static_cast<int>(padB), static_cast<int>(padL), static_cast<int>(padR));

}

void THNN_(SpatialReplicationPadding_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int padL, int padR,
           int padT, int padB) {

  THArgCheck(THCTensor_canUse32BitIndexMath(state, input), 2,
                "input tensor must fit into 32-bit index math");
  THArgCheck(THCTensor_canUse32BitIndexMath(state, gradOutput), 3,
                "output gradient tensor must fit into 32-bit index math");

  int planeDim = 0;
  int dimh = 1;
  int dimw = 2;

  int numInputDims = THCTensor_(nDimensionLegacyNoScalars)(state, input);
  if (numInputDims == 4) {
    planeDim++;
    dimh++;
    dimw++;
  }
  int iheight = input->size(dimh);
  int iwidth = input->size(dimw);
  int oheight = iheight + padT + padB;
  int owidth  = iwidth + padL + padR;

  THArgCheck(owidth == THCTensor_(size)(state, gradOutput, dimw), 3,
             "gradOutput width unexpected. Expected: %d, Got: %d",
             owidth, THCTensor_(size)(state, gradOutput, dimw));
  THArgCheck(oheight == THCTensor_(size)(state, gradOutput, dimh), 3,
             "gradOutput height unexpected. Expected: %d, Got: %d",
             oheight, THCTensor_(size)(state, gradOutput, dimh));

  THCTensor_(resizeAs)(state, gradInput, input);
  THCTensor_(zero)(state, gradInput);

  THCDeviceTensor<scalar_t, 4> devGradInput;
  THCDeviceTensor<scalar_t, 4> devGradOutput;

  if (numInputDims == 3) {
    devGradInput = toDeviceTensor<scalar_t, 3>(state, gradInput).upcastOuter<4>();
    devGradOutput = toDeviceTensor<scalar_t, 3>(state, gradOutput).upcastOuter<4>();
  } else {
    devGradInput = toDeviceTensor<scalar_t, 4>(state, gradInput);
    devGradOutput = toDeviceTensor<scalar_t, 4>(state, gradOutput);
  }

  int outputPlaneSize = devGradOutput.getSize(2) * devGradOutput.getSize(3);
  dim3 gridSize(THCCeilDiv(outputPlaneSize, 256),
            devGradOutput.getSize(1),
            devGradOutput.getSize(0));
  dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

 hipLaunchKernelGGL( SpatialReplicationPadding_updateGradInput<scalar_t>, dim3(gridSize), dim3(blockSize), 0, THCState_getCurrentStream(state), 
    devGradInput, devGradOutput, static_cast<int>(padT), static_cast<int>(padB), static_cast<int>(padL), static_cast<int>(padR));

}

#endif
