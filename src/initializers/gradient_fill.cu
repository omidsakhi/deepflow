#include "core/common_cu.h"

#include "initializers/gradient_fill.h"
#include "nodes/variable.h"

__global__
void GradientFillKernel(const int n, const int channels, const int height, const int width, float *out)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		int indexTemp = i;
		const int x = indexTemp % width;
		indexTemp /= width;
		const int y = indexTemp % height;
		indexTemp /= height;
		const int c = indexTemp % channels;
		
		out[i] = (((c == 0) ? ((float)x / (float)(width-1)) : ((float)y / (float)(height-1))) - 0.5f) * 2.0f;
	}
}

GradientFill::GradientFill(deepflow::InitParam *param) : Initializer(param) {
	LOG_IF(FATAL, param->has_gradient_fill_param() == false) << "param.has_gradient_fill_param() == false";
}

void GradientFill::apply(Node *node) {
	auto dims = node->output(0)->dims();
	LOG_IF(FATAL, dims[1] > 2) << "[FAILED] - Number of channels for gradient fill must be less than 2.";
	auto size = node->output(0)->value()->size();
	GradientFillKernel << <numOfBlocks(size), maxThreadsPerBlock >> >(size, dims[1], dims[2], dims[3], (float*)node->output(0)->value()->gpu_data(DF_LINE));
	DF_KERNEL_CHECK();
}

std::string GradientFill::to_cpp() const
{
	std::string cpp = "df.gradient_fill(";
	cpp += "{" + std::to_string(_dims[0]) + ", " + std::to_string(_dims[1]) + ", " + std::to_string(_dims[2]) + ", " + std::to_string(_dims[3]) + "}) ";	
	return cpp;
}
