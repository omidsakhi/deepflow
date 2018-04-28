#include "nodes/gabor_kernel.h"

__global__
void GaborWeightsKernel(const int n, const bool scaled, const float phi, const int window_size, const int num_orientations, const int num_scales, const float *orientations, const float *scales, float * w)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		int num_filters = num_orientations * num_scales;
		int indexTemp = i;
		const int x = indexTemp % window_size;
		indexTemp /= window_size;
		const int y = indexTemp % window_size;
		indexTemp /= window_size;
		const int output_channel = indexTemp;
		const int ori = indexTemp % num_orientations;
		indexTemp /= num_orientations;
		const int scale = indexTemp;
		float half_window_size = (float)window_size / 2.0f;
		int cx = x - half_window_size;
		int cy = y - half_window_size;
		float theta = orientations[ori];
		float sigma = scales[scale];
		float lambda = sigma / 0.35f;
		float xprime = cx * cos(theta) + cy * sin(theta);
		float yprime = -cx * sin(theta) + cy * cos(theta);
		float xprime2 = xprime * xprime;
		float yprime2 = yprime * yprime;
		float sigma2 = sigma * sigma;
		float alpha = 1;
		if (scaled)
			alpha = 0.15915494309 / sigma2;
		w[i] = alpha * exp(-0.5 * (xprime2 + yprime2) / sigma2) * cos( 2* 3.14159265358979323846f * xprime / lambda + phi);
	}
}

GaborKernel::GaborKernel(deepflow::NodeParam * param) : Node(param)
{
	LOG_IF(FATAL, param->has_gabor_kernel_param() == false) << "param.has_gabor_kernel_param() == false";
}

void GaborKernel::init()
{
	auto gparam = _param->gabor_kernel_param();	
	_apply_scale = gparam.apply_scale();
	_phi = gparam.phi();
	_num_orientations = gparam.orientations_size();
	_num_scales = gparam.scales_size();
	DF_NODE_CUDA_CHECK(cudaMalloc(&_d_orientations, _num_orientations * sizeof(float)));	
	DF_NODE_CUDA_CHECK(cudaMemcpy(_d_orientations, gparam.orientations().data(), _num_orientations * sizeof(float), cudaMemcpyHostToDevice));
	DF_NODE_CUDA_CHECK(cudaMalloc(&_d_scales, _num_scales * sizeof(float)));
	DF_NODE_CUDA_CHECK(cudaMemcpy(_d_scales, gparam.scales().data(), _num_scales * sizeof(float), cudaMemcpyHostToDevice));
	float max_scale = -FLT_MAX;
	for (int i = 0; i < _num_scales; ++i) {
		float v = gparam.scales(i);		
		if (v > max_scale)
			max_scale = v;
	}
	_window_size = ceil(4 * max_scale);
	_window_size = _window_size % 2 == 0 ? _window_size + 1 : _window_size;
	auto num_filters = _num_orientations * _num_scales;	
	std::array<int, 4> dims = { num_filters, 1, _window_size, _window_size };
	_outputs[0]->initValue(dims);
	generate();
}

void GaborKernel::forward()
{
}

void GaborKernel::backward()
{
}

std::string GaborKernel::to_cpp() const
{
	return std::string();
}

void GaborKernel::generate()
{
	auto size = _outputs[0]->value()->size();
	GaborWeightsKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, _apply_scale, _phi, _window_size, _num_orientations, _num_scales, _d_orientations, _d_scales, (float*)_outputs[0]->value()->mutableData());
	DF_KERNEL_CHECK();
}
