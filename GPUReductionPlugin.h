#ifndef GPUREDUCTIONPLUGIN_H
#define GPUREDUCTIONPLUGIN_H

#include "Plugin.h"
#include "Tool.h"
#include "PluginProxy.h"
#include <string>
#include <map>

class GPUReductionPlugin : public Plugin, public Tool {

	public:
		void input(std::string file);
		void run();
		void output(std::string file);
	private:
                std::string inputfile;
		std::string outputfile;
 //               std::map<std::string, std::string> parameters;
};


#define BLOCK_SIZE 512 //@@ You can change this


__global__ void total(float *input, float *output, int len) {
  //@@ Load a segment of the input vector into shared memory
  __shared__ float partialSum[2 * BLOCK_SIZE];
  unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
  if (start + t < len)
    partialSum[t] = input[start + t];
  else
    partialSum[t] = 0;
  if (start + BLOCK_SIZE + t < len)
    partialSum[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
  else
    partialSum[BLOCK_SIZE + t] = 0;
  //@@ Traverse the reduction tree
  for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
    __syncthreads();
    if (t < stride)
      partialSum[t] += partialSum[t + stride];
  }
  //@@ Write the computed sum of the block to the output vector at the
  //@@ correct index
  if (t == 0)
    output[blockIdx.x] = partialSum[0];
}

#endif
