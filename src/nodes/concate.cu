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
}

void Concate::init()
{
	auto d1 = _inputs[0]->dims();
	auto d2 = _inputs[1]->dims();
	LOG_IF(FATAL, d1[0] != d2[0] || d1[2] != d2[2] || d1[3] != d2[3]) << "Concate " << _name << " | inputs must have the same number of samples, width and height - " << _inputs[0]->value()->shape() << " vs " << _inputs[1]->value()->shape();
	_first_input_channels = d1[1];
	_second_input_channels = d2[1];
	_height = d1[2];
	_width = d1[3];
	_output_channels = _first_input_channels + _second_input_channels;
	_outputs[0]->initValue({ d1[0], _output_channels, d1[2], d1[3] });
	_outputs[0]->initDiff();	
}

void Concate::forward()
{
	int size;
	size = _inputs[0]->value()->size();
	ConcateKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, true, _width, _height, _first_input_channels, _output_channels, 0, (float*) _inputs[0]->value()->data(), (float*) _outputs[0]->value()->mutableData());
	DF_KERNEL_CHECK();
	size = _inputs[1]->value()->size();
	ConcateKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, true, _width, _height, _second_input_channels, _output_channels, _first_input_channels, (float *) _inputs[1]->value()->data(), (float*) _outputs[0]->value()->mutableData());
	DF_KERNEL_CHECK();
}

void Concate::backward()
{
	int size;
	if (_inputs[0]->diff()) {
		size = _inputs[0]->value()->size();
		ConcateKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, false, _width, _height, _first_input_channels, _output_channels, 0, (float*) _outputs[0]->diff()->data(), (float*) _inputs[0]->diff()->mutableData());
		DF_KERNEL_CHECK();
	}
	if (_inputs[1]->diff()) {
		size = _inputs[1]->value()->size();
		ConcateKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, false, _width, _height, _second_input_channels, _output_channels, _first_input_channels, (float *) _outputs[0]->diff()->data(), (float*) _inputs[1]->diff()->mutableData());
		DF_KERNEL_CHECK();
	}
}

std::string Concate::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.concate(" + _input_name_for_cpp(0) + ", " + _input_name_for_cpp(1) + ", ";
	cpp += "\"" + _name + "\"); ";	
	return cpp;
}
