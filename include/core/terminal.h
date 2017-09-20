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

class DeepFlowDllExport Terminal : public std::enable_shared_from_this<Terminal> {
public :
	Terminal(std::shared_ptr<Node> parentNode, int index, TerminalType type);	
	std::shared_ptr<Node> parentNode() const;		
	const TerminalType& type() const;
	virtual void connectTerminal(std::shared_ptr<Terminal> terminal) = 0;
	virtual std::array<int, 4> dims() = 0;
	virtual std::shared_ptr<Tensor> value() = 0;
	virtual std::shared_ptr<Tensor> diff() = 0;
	virtual const std::string& name() const = 0;
	const int& index() const;
protected:
	std::shared_ptr<Node> _parentNode;
	int _index = -1;		
	TerminalType _reader_type;
};

class NodeOutput;

class DeepFlowDllExport NodeInput : public Terminal {
public:
	NodeInput(std::shared_ptr<Node> parentNode, int index);	
	void connect(std::shared_ptr<NodeOutput> terminal);
	std::array<int, 4> dims();
	std::shared_ptr<Tensor> value();
	std::shared_ptr<Tensor> diff();
	virtual const std::string& name() const;
	virtual void connectTerminal(std::shared_ptr<Terminal> terminal);
	std::shared_ptr<Node> connectedNode() const;
	std::shared_ptr<Terminal> connectedTerminal() const;
protected:
	std::shared_ptr<Terminal> _connected_terminal;
};

class DeepFlowDllExport NodeOutput : public Terminal {
public:
	NodeOutput(std::shared_ptr<Node> parentNode, int index, const std::string &name);		
	void initValue(std::array<int, 4> dims, Tensor::TensorType type = Tensor::Float);	
	std::array<int, 4> dims();
	void feed(std::shared_ptr<NodeOutput> t);
	void initDiff();
	void resetDiff();
	void resetValue();
	void cpyValueToDiff();
	std::shared_ptr<Tensor> value();
	std::shared_ptr<Tensor> diff();
	virtual const std::string& name() const;
	virtual void connectTerminal(std::shared_ptr<Terminal> terminal);
	virtual std::set<std::shared_ptr<Node>> connectedNodes() const;
	virtual std::set<std::shared_ptr<Terminal>> connectedTerminals() const;
protected:
	std::set<std::shared_ptr<Terminal>> _connected_terminals;
	std::shared_ptr<Tensor> _value;
	std::shared_ptr<Tensor> _diff;
	std::string _name;
};

using NodeOutputPtr = std::shared_ptr<NodeOutput>;

using NodeInputPtr = std::shared_ptr<NodeInput>;
