#pragma once

#include "core/node.h"

class DeepFlowDllExport Switch : public Node {
public:
	Switch(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs();
	std::list<std::shared_ptr<Node>> outputNodes() const;
	std::list<std::shared_ptr<Node>> inputNodes() const;
	void init();
	void forward();
	void backward();
	std::string to_cpp() const;
	void setEnabled(bool state);
private:
	bool m_on = true;
};
