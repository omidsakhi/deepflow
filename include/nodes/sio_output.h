#pragma once

#include "core/node.h"

#include "src/sio_client.h"

class DeepFlowDllExport SIOOutput : public Node {
public:
	SIOOutput(const deepflow::NodeParam &param);
	int minNumInputs();
	int minNumOutputs() { return 0; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	std::string to_cpp() const;
	ForwardType forwardType() { return ALWAYS_FORWARD; }
	BackwardType backwardType() { return NEVER_BACKWARD; }
private:
	void on_connected();
	void on_close(sio::client::close_reason const& reason);
	void on_fail();
private:
	int _num_inputs = 0;	
	deepflow::ActionTime _print_time;
	std::string _host;
	int _port;
	sio::client _client;			
	sio::socket::ptr _socket = nullptr;
};

