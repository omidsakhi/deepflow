#include "nodes/concate.h"

__global__ void ConcateKernel(
	const int size,
	const bool forward,
	const int width,
	const int height,
	const int input_channels,
	const int output_channels,
	const int channel_offset,
	const float* x_dy,
	float* y_dx) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < size) {
		int temp = i;
		const int w = temp % width;
		temp /= width;
		const int h = temp % height;		
		int tempi = temp / height;		
		const int ci = tempi % input_channels;
		const int ni = tempi / input_channels;
		int iindex = ni * (input_channels * width * height) + ci * (width * height) + h * width + w;
		int oindex = ni * (output_channels * width * height) + (ci + channel_offset) * (width * height) + h * width + w;
		if (forward)
			y_dx[oindex] = x_dy[iindex];
		else
			y_dx[iindex] = x_dy[oindex];
	}
}

Concate::Concate(deepflow::NodeParam * param) : Node(param)
{
	LOG_IF(FATAL, param->has_concate_param() == false) << "param.has_concate_param() == false";
	_num_inputs = param->concate_param().num_inputs();
}

int Concate::minNumInputs()
{
	return _num_inputs;
}

void Concate::init()
{	
	auto firstInputDims = _inputs[0]->dims();
	_height = firstInputDims[2];
	_width = firstInputDims[3];
	auto num_samples = firstInputDims[0];
	_output_channels = firstInputDims[1];
	for (int i = 1; i < _num_inputs; ++i) {
		auto input = _inputs[i];
		auto inputDims = input->dims();
		LOG_IF(FATAL, inputDims[0] != firstInputDims[0] || 
			inputDims[2] != firstInputDims[2] || 
			inputDims[3] != firstInputDims[3]) << "Concate " << _name << " | inputs must have the same number of samples, width and height as " << _inputs[0]->value()->shape() << " but input " << i << " has a shape of " << " vs " << input->value()->shape();
		_output_channels += inputDims[1];
	}	
	_outputs[0]->initValue({ num_samples, _output_channels, _height, _width });
	_outputs[0]->initDiff();	
}

void Concate::forward()
{
	int channel_offset = 0;
	for (int i = 0; i < _num_inputs; ++i) {
		auto input = _inputs[i];
		int size = input->value()->size();
		int channels = input->value()->dim(1);
		ConcateKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, true, _width, _height, channels, _output_channels, channel_offset, input->value()->gpu_data(DF_LINE), _outputs[0]->value()->gpu_data(DF_LINE));
		DF_KERNEL_CHECK();
		channel_offset += channels;
	}	
}

void Concate::backward()
{	
	int channel_offset = 0;
	for (int i = 0; i < _num_inputs; ++i) {
		auto input = _inputs[i];
		int size = input->value()->size();
		int channels = input->value()->dim(1);
		if (input->diff()) {
			ConcateKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, false, _width, _height, channels, _output_channels, channel_offset, _outputs[0]->diff()->gpu_data(DF_LINE), input->diff()->gpu_data(DF_LINE));
			DF_KERNEL_CHECK();
		}
		channel_offset += channels;
	}
}

std::string Concate::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.concate(" + _input_name_for_cpp(0) + ", " + _input_name_for_cpp(1) + ", ";
	cpp += "\"" + _name + "\"); ";	
	return cpp;
}
