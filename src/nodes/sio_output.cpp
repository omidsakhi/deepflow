#include "nodes/sio_output.h"

SIOOutput::SIOOutput(const deepflow::NodeParam & param) : Node(param)
{
	LOG_IF(FATAL, param.has_sio_output_param() == false) << "param.has_sio_output_param() == false";
	auto sio_param = _param.sio_output_param();
	_num_inputs = sio_param.num_inputs();
	_print_time = sio_param.print_time();
	_host = sio_param.host();
	_port = sio_param.port();
}

int SIOOutput::minNumInputs()
{
	return _num_inputs;
}

void SIOOutput::initForward()
{
	_client.set_open_listener(std::bind(&SIOOutput::on_connected, this));
	_client.set_close_listener(std::bind(&SIOOutput::on_close, this, std::placeholders::_1));
	_client.set_fail_listener(std::bind(&SIOOutput::on_fail, this));
	std::string url = "http://" + _host + ":" + std::to_string(_port);
	LOG(INFO) << "CONNECTING TO " << url;
	_client.connect(url);
}

void SIOOutput::initBackward()
{
}

void SIOOutput::forward()
{
	if (_socket) {
		auto outgoing_message = sio::object_message::create();
		outgoing_message->get_map()["node"] = sio::string_message::create(_name);
		auto input_array = sio::array_message::create();				
		for (int i = 0; i < _inputs.size(); ++i) {
			auto node_input = _inputs[i];
			auto connected_node = node_input->connectedNode();
			if (connected_node) {
				auto input_node_value = node_input->value();
				auto input_node_value_size = input_node_value->size();								
				
				auto input_element = sio::object_message::create();				
				input_element->get_map()["name"] = sio::string_message::create(connected_node->name());

				auto shape_array = sio::array_message::create();				
				auto dims = input_node_value->dims();
				for (int i = 0; i < 4; i++)
					shape_array->get_vector().push_back(sio::int_message::create(dims[i]));
				input_element->get_map()["shape"] = shape_array;

				auto data_array = sio::array_message::create();				
				auto data = input_node_value->cpyToHost<float>();
				for (int i = 0; i < input_node_value_size; i++)
					data_array->get_vector().push_back(sio::double_message::create(data->at(i)));
				input_element->get_map()["data"] = data_array;

				input_array->get_vector().push_back(input_element);
			}
		}		
		outgoing_message->get_map()["inputs"] = input_array;
		_socket->emit("session-data", outgoing_message);
	}
}

void SIOOutput::backward()
{
	LOG(FATAL);
}

std::string SIOOutput::to_cpp() const
{
	return std::string();
}

void SIOOutput::on_connected()
{
	LOG(INFO) << "CONNECTED: " << _name;
	_socket = _client.socket();
}

void SIOOutput::on_close(sio::client::close_reason const & reason)
{
	LOG(INFO) << "CONNECTION CLOSED: " << _name;
	_socket = nullptr;
}

void SIOOutput::on_fail()
{
	LOG(INFO) << "CONNECTION FAILED: " << _name;
	_socket = nullptr;
}
