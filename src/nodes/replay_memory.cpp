#include "nodes/replay_memory.h"

ReplayMemory::ReplayMemory(const deepflow::NodeParam & param)
{
	LOG_IF(FATAL, param.has_replay_memory_param() == false) << "param.has_replay_memory_param() == false";
}

void ReplayMemory::initForward()
{
	auto capacity = _param.replay_memory_param().capacity();
	auto inputDims = _inputs[0]->value()->dims();
	_batch_size = inputDims[0];
	LOG_IF(FATAL, capacity % _batch_size != 0) << "Memory capacity must be dividable by the batch size.";
	size_t mem_size = capacity * inputDims[1] * inputDims[2] * inputDims[3];
	DF_NODE_CUDA_CHECK(cudaMalloc(&dev_memory, mem_size * sizeof(float)));
}

void ReplayMemory::initBackward()
{
}

void ReplayMemory::forward()
{	
	size_t n = _inputs[0]->value()->size();
	cpy(n, 1.0f, _inputs[0]->value()->data(), 0.0f, dev_memory + _input_head);
	if ((_input_head + _batch_size) >= _capacity) {
		_input_head = 0;
		_output_head = rand() % (_capacity - _batch_size);
	}
	else {
		_output_head = rand() % _input_head;
		_input_head += _batch_size;		
	}
	cpy(n, 1.0, dev_memory + _output_head, 0.0f, _outputs[0]->value()->mutableData());
}

void ReplayMemory::backward()
{
	LOG(FATAL);
}

std::string ReplayMemory::to_cpp() const
{
	return std::string();
}
