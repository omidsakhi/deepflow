#pragma once

#include "core/export.h"
#include "core/common_cu.h"

#include <string>
#include <iostream>

class DeepFlowDllExport CudaHelper {
public:
	CudaHelper();
	std::string cudaStatusToString(cudnnStatus_t status);
	static void setOptimalThreadsPerBlock();
	static size_t numOfBlocks(const size_t &size);
	static int maxThreadsPerBlock;
protected:
	static std::vector<cudaDeviceProp> _cudaDeviceProps;	
};