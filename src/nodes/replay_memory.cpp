#include "nodes/replay_memory.h"

ReplayMemory::ReplayMemory(deepflow::NodeParam *param) : Node(param)
{
	LOG_IF(FATAL, param->has_replay_memory_param() == false) << "param.has_replay_memory_param() == false";
}

void ReplayMemory::init()
{
	_capacity = _param->replay_memory_param().capacity();
	auto inputDims = _inputs[0]->value()->dims();
	_num_samples_per_batch = inputDims[0];
	LOG_IF(FATAL, _capacity % _num_samples_per_batch != 0) << "Memory capacity must be dividable by the batch size.";
	_size_per_sample = inputDims[1] * inputDims[2] * inputDims[3];
	_mem_size = _capacity * _size_per_sample;
	DF_NODE_CUDA_CHECK(cudaMalloc(&dev_memory, _mem_size * sizeof(float)));
	_outputs[0]->initValue(inputDims);	
}

void ReplayMemory::forward()
{	
	size_t n = _inputs[0]->value()->size();
	LOG_IF(INFO, _verbose > 3) << " INPUT HEAD: " << _input_head;
	cpy(n, 1.0f, _inputs[0]->value()->gpu_data(), 0.0f, dev_memory + _input_head);
	if ((_input_head + _num_samples_per_batch * _size_per_sample) >= _mem_size) {
		_input_head = 0;
		_available_samples = _capacity;
	}
	else {		
		_input_head += _num_samples_per_batch * _size_per_sample;
		_available_samples += _num_samples_per_batch;
		_available_samples = (_available_samples >= _capacity) ? _capacity : _available_samples;
	}
	_output_head = (rand() % (_available_samples - _num_samples_per_batch + 1)) * _size_per_sample;
	LOG_IF(INFO, _verbose > 3) << " OUTPUT HEAD: " << _output_head;
	cpy(n, 1.0, dev_memory + _output_head, 0.0f, _outputs[0]->value()->gpu_data());
}

void ReplayMemory::backward()
{
	
}

std::string ReplayMemory::to_cpp() const
{
	return std::string();
}
