// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include "GPUReductionPlugin.h"

void GPUReductionPlugin::input(std::string infile) {
 readParameterFile(infile);
}

void GPUReductionPlugin::run() {}

void GPUReductionPlugin::output(std::string outfile) {

  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  numInputElements = atoi(myParameters["N"].c_str());

  hostInput = (float*) malloc (numInputElements*sizeof(float));
   std::ifstream myinput((std::string(PluginManager::prefix())+myParameters["data"]).c_str(), std::ios::in);
 int i;
 for (i = 0; i < numInputElements; ++i) {
        int k;
        myinput >> k;
        hostInput[i] = k;
 }



  numOutputElements = numInputElements / (BLOCK_SIZE << 1);
  if (numInputElements % (BLOCK_SIZE << 1)) {
    numOutputElements++;
  }
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  //@@ Allocate GPU memory here
  cudaMalloc(&deviceInput, sizeof(float) * numInputElements);
  cudaMalloc(&deviceOutput, sizeof(float) * numOutputElements);

  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, sizeof(float) * numInputElements,
             cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(numOutputElements, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  //@@ Launch the GPU Kernel here
  total<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput,
                               numInputElements);

  cudaDeviceSynchronize();
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * numOutputElements,
             cudaMemcpyDeviceToHost);


  /********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input. For simplicity, we do not
   * require that for this lab.
   ********************************************************************/
  for (int ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }

  std::cout << hostOutput[0] << std::endl;
  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceOutput);


  free(hostInput);
  free(hostOutput);

}


PluginProxy<GPUReductionPlugin> GPUReductionPluginProxy = PluginProxy<GPUReductionPlugin>("GPUReduction", PluginManager::getInstance());

