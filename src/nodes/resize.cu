#include "nodes/resize.h"
#include "core/common_cu.h"


__global__ void NearestNeighborKernel(
	const int y_size,
	const int num_channels,
	const int input_height,
	const int input_width,
	const int output_height,
	const int output_width,
	const float height_scale,
	const float width_scale,
	const float* X,
	float* Y) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < y_size) {
		int indexTemp = i;
		const int w = indexTemp % output_width;
		indexTemp /= output_width;
		const int h = indexTemp % output_height;
		indexTemp /= output_height;
		const int c = indexTemp % num_channels;
		indexTemp /= num_channels;
		const int n = indexTemp;

		const int in_y = fminf(h / height_scale, input_height - 1);
		const int in_x = fminf(w / width_scale, input_width - 1);
		Y[i] =
			X[((n * num_channels + c) * input_height + in_y) * input_width + in_x];
	}
}

__global__ void NearestNeighborGradientKernel(
	const int dy_size,
	const int num_channels,
	const int input_height,
	const int input_width,
	const int output_height,
	const int output_width,
	const float height_scale,
	const float width_scale,
	const float* dY,
	float* dX) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < dy_size) {
		int indexTemp = i;
		const int x = indexTemp % input_width;
		indexTemp /= input_width;
		const int y = indexTemp % input_height;
		indexTemp /= input_height;
		const int c = indexTemp % num_channels;
		indexTemp /= num_channels;
		const int n = indexTemp;

		const int out_y = fminf(y / height_scale, output_height - 1);
		const int out_x = fminf(x / width_scale, output_width - 1);
		const int out_index =
			((n * num_channels + c) * output_height + out_y) * output_width + out_x;
		
#if __CUDA_ARCH__ >= 350
		atomicAdd(dX + out_index, __ldg(dY + i));
#else 
		atomicAdd(dX + out_index, *(dY + i));
#endif
	
	}
}

Resize::Resize(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_resize_param() == false) << "param.has_resize_param() == false";
}

void Resize::initForward() {
	auto param = _param->resize_param();
	m_height_scale = param.height_scale();
	m_width_scale = param.width_scale();
	auto input_dims = _inputs[0]->value()->dims();
	_outputs[0]->initValue({input_dims[0], input_dims[1], (int) (input_dims[2] * m_height_scale), (int) (input_dims[3] * m_width_scale) });
	LOG(INFO) << "Resize " << _name << " - " << _outputs[0]->value()->shape();
}

void Resize::initBackward() {
	_outputs[0]->initDiff();
}

void Resize::forward() {
	auto size = _outputs[0]->value()->size();
	auto dims = _inputs[0]->value()->dims();
	NearestNeighborKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (
		size,
		dims[1],
		dims[2],
		dims[3],
		dims[2] * m_height_scale,
		dims[3] * m_width_scale,
		m_height_scale,
		m_width_scale,
		(float*)_inputs[0]->value()->data(),
		(float*)_outputs[0]->value()->mutableData());
	DF_KERNEL_CHECK();
}

void Resize::backward() {
	auto size = _outputs[0]->value()->size();
	auto output_dims = _outputs[0]->value()->dims();
	auto input_dims = _inputs[0]->value()->dims();
	DF_CUDA_CHECK(cudaMemset(_inputs[0]->diff()->mutableData(), 0, _inputs[0]->diff()->sizeInBytes()));
	NearestNeighborGradientKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (
		size,
		output_dims[1],
		output_dims[2],
		output_dims[3],
		input_dims[2],
		input_dims[3],
		m_height_scale,
		m_width_scale,
		(float*)_outputs[0]->diff()->data(),
		(float*)_inputs[0]->diff()->mutableData());
	DF_KERNEL_CHECK();
}

std::string Resize::to_cpp() const
{
	return "";
}