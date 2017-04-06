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
	Terminal(std::shared_ptr<Node> parentNode, int index, TerminalType type);	
	std::shared_ptr<Node> node() const;
	std::shared_ptr<Node> connectedNode() const;
	const TerminalType& type() const;
	virtual std::array<int, 4> dims() = 0;
	virtual std::shared_ptr<Tensor> value() = 0;
	virtual std::shared_ptr<Tensor> diff() = 0;
	virtual const std::string& name() const = 0;
	const int& index() const;
protected:
	std::shared_ptr<Node> _parentNode;
	int _index;	
	std::shared_ptr<Terminal> _terminal;
	TerminalType _type;
};

class OutputTerminal;

class DeepFlowDllExport InputTerminal : public Terminal {
public:
	InputTerminal(std::shared_ptr<Node> parentNode, int index);	
	void connect(std::shared_ptr<OutputTerminal> terminal);
	std::array<int, 4> dims();
	std::shared_ptr<Tensor> value();
	std::shared_ptr<Tensor> diff();
	virtual const std::string& name() const;
};

class DeepFlowDllExport OutputTerminal : public Terminal {
public:
	OutputTerminal(std::shared_ptr<Node> parentNode, int index, const std::string &name);		
	void initValue(std::array<int, 4> dims, Tensor::TensorType type = Tensor::Float);
	std::array<int, 4> dims();
	void feed(std::shared_ptr<OutputTerminal> t);
	void initDiff();	
	void cpyValueToDiff();
	std::shared_ptr<Tensor> value();
	std::shared_ptr<Tensor> diff();
	virtual const std::string& name() const;
protected:
	std::shared_ptr<Tensor> _value;
	std::shared_ptr<Tensor> _diff;
	std::string _name;
};