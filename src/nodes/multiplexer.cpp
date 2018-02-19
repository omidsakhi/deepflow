#include "nodes/multiplexer.h"

Multiplexer::Multiplexer(deepflow::NodeParam *param) : Node(param)
{
	LOG_IF(FATAL, param->has_multiplexer_param() == false) << "param.has_multiplexer_param() == false";
	auto multiplexer_param = _param->multiplexer_param();
	_num_inputs = multiplexer_param.num_inputs();
	LOG_IF(FATAL, _num_inputs == 0) << "_num_inputs == 0";
}

int Multiplexer::minNumInputs()
{
	return _num_inputs;
}

void Multiplexer::init()
{
	auto firstInputDim = _inputs[0]->value()->dims();
	for (int i = 1; i < _num_inputs; ++i) {
		auto dims = _inputs[i]->value()->dims();
		LOG_IF(FATAL, dims != firstInputDim) << _name << " Mismatch size for input " << i << " " << _inputs[0]->value()->shape() << " != " << _inputs[i]->value()->shape();
	}
	_outputs[0]->initValue(firstInputDim);
	_output_size_in_bytes = _outputs[0]->value()->sizeInBytes();
	_outputs[0]->initDiff();
	LOG(INFO) << "Multiplexer " << _name << " - " << _outputs[0]->value()->shape();
}

void Multiplexer::forward()
{	
	if (_selected_input == -1) {
		LOG_IF(INFO, _verbose > 2) << _name << "MULTIPLEXER OFF";
		return;
	}
	LOG_IF(FATAL, _selected_input >= _num_inputs) << _name << " INPUT TO SELECTOR MUST BE LESS THAN " << (_num_inputs - 1);
	LOG_IF(INFO, _verbose > 2) << "MULTIPLEXER FORWARD " << _name << " - SELECTED INPUT " << _inputs[_selected_input]->connectedNode()->name();
	DF_NODE_CUDA_CHECK(cudaMemcpy(_outputs[0]->value()->mutableData(), _inputs[_selected_input]->value()->data(), _output_size_in_bytes, cudaMemcpyDeviceToDevice))
}

void Multiplexer::backward()
{	
	if (_selected_input == -1) {
		LOG_IF(INFO, _verbose > 2) << _name << " MULTIPLEXER OFF";
		for (auto input : _inputs) {
			if (input->diff())
				fill(input->diff()->size(), 0, input->diff()->mutableData());
		}
		return;
	}
	auto input = _inputs[_selected_input];
	if (input->connectedNode() && input->diff() && input->diff()->mutableData()) {
		LOG_IF(INFO, _verbose > 2) << "MULTIPLEXER BACKWARD " << _name << " - SELECTED INPUT " << _inputs[_selected_input]->connectedNode()->name();
		cpy(_outputs[0]->diff()->size(), 1, _outputs[0]->diff()->data(), 0, input->diff()->mutableData());
	}
}

std::list<std::shared_ptr<Node>> Multiplexer::inputNodes() const
{
	std::list<std::shared_ptr<Node>> list;
	LOG_IF(FATAL, _selected_input >= _num_inputs) << _name << " INPUT TO SELECTOR MUST LESS THAN " << (_num_inputs - 1);
	if (_selected_input == -1)
		return list;
	else
		list.push_back(_inputs[_selected_input]->connectedNode());
	return list;
}

std::list<std::shared_ptr<Node>> Multiplexer::outputNodes() const
{
	std::list<std::shared_ptr<Node>> list;
	LOG_IF(FATAL, _selected_input >= _num_inputs) << _name << " INPUT TO SELECTOR MUST LESS THAN " << (_num_inputs - 1);
	if (_selected_input == -1)
		return list;
	else {
		for (auto output : _outputs) {
			for (auto node : output->connectedNodes())
				list.push_back(node);
		}
	}
	return list;
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

void Multiplexer::selectInput(int input)
{
	_selected_input = input;
}
