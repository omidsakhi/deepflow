#include "nodes/block.h"

#include <google/protobuf/text_format.h>

#include <fstream>

Block::Block()
{	
	_block_param = _param.mutable_block_param();
}

Block::Block(deepflow::NodeParam & param) : Node(param)
{
	LOG_IF(FATAL, param.has_block_param() == false) << "param.has_block_param() == false";	
	_block_param = param.mutable_block_param();
}

deepflow::NodeParam* Block::find_node_by_name(const std::string & name) const
{	
	for (int i = 0; i < _block_param->node_size(); ++i) {
		if (_block_param->node(i).name() == name) {
			return _block_param->mutable_node(i);
		}
	}
	return 0;
}

int Block::find_node_index_by_name(const std::string & name) const
{
	for (int i = 0; i < _block_param->node_size(); ++i) {
		if (_block_param->node(i).name() == name) {
			return i;
		}
	}
	return -1;
}

deepflow::SolverParam* Block::find_solver_by_name(const std::string & name) const
{
	for (int i = 0; i < _block_param->solver_size(); ++i) {
		if (_block_param->solver(i).name() == name) {
			return _block_param->mutable_solver(i);
		}
	}
	return 0;
}

deepflow::InitParam * Block::find_initializer_by_name(const std::string & name) const
{
	for (int i = 0; i < _block_param->initializer_size(); ++i) {
		if (_block_param->initializer(i).name() == name) {
			return _block_param->mutable_initializer(i);
		}
	}
	return 0;
}

deepflow::NodeParam* Block::find_node_by_output_name(const std::string & output_name) const
{
	for (int i = 0; i < _block_param->node_size(); ++i) {
		for (auto output : _block_param->node(i).output())
			if (output == output_name) {
				return _block_param->mutable_node(i);
			}
	}
	return 0;
}

std::list<deepflow::NodeParam*> Block::find_nodes_by_input_name(const std::string & input_name) const
{
	auto list = std::list<deepflow::NodeParam*>();
	for (int i = 0; i < _block_param->node_size(); ++i) {
		for (auto input : _block_param->node(i).input())
			if (input == input_name) {
				list.push_back(_block_param->mutable_node(i));
			}
	}
	return list;
}

std::string Block::get_unique_node_name(const std::string & prefix) const
{
	if (find_node_by_name(prefix) == 0)
		return prefix;
	int index = 1;
	std::string nodeName = prefix + "_" + std::to_string(index);
	while (find_node_by_name(nodeName) != 0) {
		index++;
		nodeName = prefix + "_" + std::to_string(index);
	}
	return nodeName;
}

std::string Block::get_unique_solver_name(const std::string & prefix) const
{	
	if (find_solver_by_name(prefix) == 0)
		return prefix;
	int index = 1;
	std::string nodeName = prefix + "_" + std::to_string(index);
	while (find_solver_by_name(nodeName) != 0) {
		index++;
		nodeName = prefix + "_" + std::to_string(index);
	}
	return nodeName;
}

std::string Block::get_unique_initializer_name(const std::string & prefix) const
{
	if (find_initializer_by_name(prefix) == 0)
		return prefix;
	int index = 1;
	std::string initName = prefix + "_" + std::to_string(index);
	while (find_initializer_by_name(initName) != 0) {
		index++;
		initName = prefix + "_" + std::to_string(index);
	}
	return initName;
}

void Block::replace_nodes_input(const std::string & search, const std::string & replace, const std::initializer_list<deepflow::NodeParam *> exclude_nodes)
{
	auto affected_input_nodes = find_nodes_by_input_name(search);
	for (auto node : exclude_nodes)
		affected_input_nodes.remove(node);
	for (auto affected_input_node : affected_input_nodes)
		for (int i = 0; i < affected_input_node->input_size(); i++) {
			if (affected_input_node->input(i) == search) {
				affected_input_node->set_input(i, replace);
			}
		}
}

void RemoveFromRepeatedField(
	const google::protobuf::Reflection *reflection,
	const google::protobuf::FieldDescriptor *field,
	google::protobuf::Message *message,
	int row,
	int count)
{
	int size = reflection->FieldSize(*message, field);
	for (int i = row; i < size - count; ++i)
		reflection->SwapElements(message, field, i, i + count);
	for (int i = 0; i < count; ++i)
		reflection->RemoveLast(message, field);
}

void Block::remove_node_by_name(const std::string & name)
{
	deepflow::NodeParam *node = find_node_by_name(name);
	LOG_IF(FATAL, node == nullptr) << "Failed to find node with name " << name;
	int num_inputs = node->input_size();
	int num_outputs = node->output_size();
	LOG_IF(FATAL, num_outputs > num_inputs) << "Failed to remove node " << node->name() << "due to num_outputs > num_inputs";
	for (int i = 0; i < num_outputs; ++i)
		replace_nodes_input(node->output(i), node->input(i), {});	
	RemoveFromRepeatedField(_block_param->GetReflection(), _block_param->GetDescriptor()->FindFieldByName("node"), node, find_node_index_by_name(node->name())  , 1);
}

void Block::save_as_binary(std::string file_path)
{	
	std::fstream output(file_path, std::ios::out | std::ios::trunc | std::ios::binary);
	LOG_IF(FATAL, !_block_param->SerializeToOstream(&output)) << "Failed to write block to " << file_path;
	output.close();
}

void Block::save_as_text(std::string file_path)
{	
	std::string text;
	google::protobuf::TextFormat::PrintToString(*_block_param, &text);
	std::ofstream out(file_path);
	out << text;
	out.close();
}

deepflow::NodeParam* Block::add_node()
{
	return _block_param->add_node();
}

deepflow::PhaseParam * Block::add_phase()
{
	return _block_param->add_phase();
}

deepflow::SolverParam * Block::add_solver()
{
	return _block_param->add_solver();
}

deepflow::InitParam * Block::add_initializer()
{
	return _block_param->add_initializer();
}

void Block::print_nodes()
{
	for (auto node : _block_param->node()) {
		std::string inputs = (node.input_size() > 0) ? node.input(0) : "";
		for (int i = 1; i < node.input_size(); ++i)
			inputs += ", " + node.input(i);
		std::string outputs = (node.output_size() > 0) ? node.output(0) : "";
		for (int i = 1; i < node.output_size(); ++i)
			outputs += ", " + node.output(i);
		if (!outputs.empty())
			outputs = " -> " + outputs;
		LOG(INFO) << "Node " << node.name() << "(" << inputs << ")" << outputs;
	}
}

void Block::print_phases()
{
	for (auto phase : _block_param->phase()) {
		LOG(INFO) << phase.phase() << " <-> " << deepflow::PhaseParam_PhaseBehaviour_Name(phase.behaviour());
	}
}

void Block::load_from_binary(std::string file_path)
{
	std::fstream input(file_path, std::ios::in | std::ios::binary);
	LOG_IF(FATAL, !_block_param->ParseFromIstream(&input)) << "Failed to read binary block from "  << file_path;
	input.close();
}

google::protobuf::RepeatedPtrField<deepflow::PhaseParam> Block::phases()
{
	return _block_param->phase();
}

google::protobuf::RepeatedPtrField<deepflow::NodeParam> Block::nodes()
{
	return _block_param->node();
}

google::protobuf::RepeatedPtrField<deepflow::SolverParam> Block::solvers()
{
	return _block_param->solver();
}

google::protobuf::RepeatedPtrField<deepflow::InitParam> Block::initializers()
{
	return _block_param->initializer();
}

int Block::minNumInputs()
{
	return 0;
}

int Block::minNumOutputs()
{
	return 0;
}

void Block::initForward()
{
}

void Block::initBackward()
{
}

void Block::forward()
{
}

void Block::backward()
{
}

std::string Block::to_cpp() const
{
	return std::string();
}
