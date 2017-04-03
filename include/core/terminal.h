#pragma once

#include "core/export.h"
#include "core/tensor.h"

#include <memory>
#include <vector>

class Node;

enum TerminalType {
	Input,
	Output
}; 

class DeepFlowDllExport Terminal {
public :
	Terminal(std::shared_ptr<Node> parentNode, int index,  std::string name, TerminalType type);	
	std::shared_ptr<Node> parentNode();
	std::shared_ptr<Node> otherNode();
	TerminalType type();
	virtual std::array<int, 4> dims() = 0;
	virtual std::shared_ptr<Tensor> value() = 0;
	virtual std::shared_ptr<Tensor> diff() = 0;
protected:
	std::shared_ptr<Node> _parentNode;
	int _index;
	std::string _name;
	std::shared_ptr<Terminal> _terminal;
	TerminalType _type;
};

class OutputTerminal;

class DeepFlowDllExport InputTerminal : public Terminal {
public:
	InputTerminal(std::shared_ptr<Node> parentNode, int index, std::string name);	
	void connect(std::shared_ptr<OutputTerminal> terminal);
	std::array<int, 4> dims();
	std::shared_ptr<Tensor> value();
	std::shared_ptr<Tensor> diff();
};

class DeepFlowDllExport OutputTerminal : public Terminal {
public:
	OutputTerminal(std::shared_ptr<Node> parentNode, int index, std::string name);		
	void initValue(std::array<int, 4> dims, Tensor::TensorType type = Tensor::Float);
	std::array<int, 4> dims();		
	void initDiff();	
	void cpyValueToDiff();
	std::shared_ptr<Tensor> value();
	std::shared_ptr<Tensor> diff();
protected:
	std::shared_ptr<Tensor> _value;
	std::shared_ptr<Tensor> _diff;
};