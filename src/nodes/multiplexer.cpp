#include "nodes/multiplexer.h"

Multiplexer::Multiplexer(const deepflow::NodeParam & param) : Node(param)
{
	LOG_IF(FATAL, param.has_multiplexer_param() == false) << "param.has_multiplexer_param() == false";
	auto multiplexer_param = _param.multiplexer_param();
	_num_inputs = multiplexer_param.num_inputs();
	LOG_IF(FATAL, _num_inputs == 0) << "_num_inputs == 0";
}

int Multiplexer::minNumInputs()
{
	return _num_inputs + 1; // including the selector
}

void Multiplexer::initForward()
{
	auto firstInputDim = _inputs[0]->value()->dims();
	for (int i = 1; i < _num_inputs; ++i) {
		auto dims = _inputs[i]->value()->dims();
		LOG_IF(FATAL, dims != firstInputDim) << "Mismatch size for input " << i << " " << _inputs[0]->value()->shape() << " != " << _inputs[i]->value()->shape();
	}
	LOG_IF(FATAL, _inputs[_num_inputs]->connectedNode() == nullptr) << "Selector is not connected";
	_outputs[0]->initValue(firstInputDim);
	_output_size_in_bytes = _outputs[0]->value()->sizeInBytes();
	LOG(INFO) << "Initializing Multiplexer " << _name << " - " << _outputs[0]->value()->shape();
}

void Multiplexer::initBackward()
{
	_outputs[0]->initDiff();
}

void Multiplexer::forward()
{
	float tmp = _inputs[_num_inputs]->value()->toFloat();
	LOG_IF(WARNING, floor(tmp) != tmp) << "Input to selector must be an integer value and not " << tmp;
	_selected_input = floor(tmp);
	LOG_IF(FATAL, _selected_input >= _num_inputs) << "Input to selector must be between 0 and " << (_num_inputs - 1);
	DF_NODE_CUDA_CHECK(cudaMemcpy(_outputs[0]->value()->mutableData(), _inputs[_selected_input]->value()->data(), _output_size_in_bytes, cudaMemcpyDeviceToDevice))
}

void Multiplexer::backward()
{	
	int tmp = floor(_inputs[_num_inputs]->value()->toFloat());
	LOG_IF(FATAL, _selected_input != tmp) << "Input to selector changed between forward and backward passes.";
	auto input = _inputs[_selected_input];
	if (input->connectedNode()->propagateBack()) {
		cpy(_outputs[0]->diff()->size(), 1, _outputs[0]->diff()->data(), 1, input->diff()->mutableData());
	}
}

std::string Multiplexer::to_cpp() const
{
	std::string inputs = _input_name_for_cpp(0);
	for (int i = 1; i < _num_inputs; ++i) {
		inputs += ", " + _input_name_for_cpp(i);
	}

	std::string cpp = "df.multiplexer( {" + inputs + "} , ";
	cpp += _input_name_for_cpp(_num_inputs) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
