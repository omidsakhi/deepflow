#pragma once

#include "core/node.h"

#include "src/sio_client.h"

class DeepFlowDllExport SIOOutput : public Node {
public:
	SIOOutput(deepflow::NodeParam *param);
	int minNumInputs();
	int minNumOutputs() { return 0; }
	std::string op_name() const override { return "sio_output"; }
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
private:
	void on_connected();
	void on_close(sio::client::close_reason const& reason);
	void on_fail();
private:
	int _num_inputs = 0;	
	std::string _host;
	int _port;
	sio::client _client;			
	sio::socket::ptr _socket = nullptr;
};

