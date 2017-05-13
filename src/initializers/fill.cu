#include "core/common_cu.h"

#include "initializers/fill.h"
#include "nodes/variable.h"

__global__
void FillKernel(const int n, float *a, const float v)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) a[i] = v;
}

Fill::Fill(const deepflow::InitParam &_block_param) : Initializer(_block_param) {
	LOG_IF(FATAL, _block_param.has_fill_param() == false) << "param.has_fill_param() == false";		
}

void Fill::apply(Variable *variable) {
	float value = _param.fill_param().value();
	auto size = variable->output(0)->value()->size();	
	FillKernel <<< numOfBlocks(size), maxThreadsPerBlock >>> (size, (float*)variable->output(0)->value()->mutableData(), value);
	DF_KERNEL_CHECK();
}

std::string Fill::to_cpp() const
{	
	float value = _param.fill_param().value();
	std::string op;
	bool omit_value = true;
	if (value == 0)
		op = "zeros";
	else if (value == 1)
		op = "ones";
	else {
		op = "fill";
		omit_value = false;
	}
	std::string cpp = "df."+op+"(";
	cpp += "{" + std::to_string(_dims[0]) + ", " + std::to_string(_dims[1]) + ", " + std::to_string(_dims[2]) + ", " + std::to_string(_dims[3]) + "}, ";
	if (!omit_value)
		cpp += std::to_string(value);
	cpp += ")";
	return cpp;
}
