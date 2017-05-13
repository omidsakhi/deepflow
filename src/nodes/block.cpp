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

std::shared_ptr<deepflow::NodeParam> Block::find_node_by_name(const std::string & name) const
{
	auto _block_param = _param.block_param();
	for (int i = 0; i < _block_param.node_size(); ++i) {
		if (_block_param.node(i).name() == name) {
			return std::shared_ptr<deepflow::NodeParam>(_block_param.mutable_node(i));
		}
	}
	return 0;
}

std::shared_ptr<deepflow::SolverParam> Block::find_solver_by_name(const std::string & name) const
{
	for (int i = 0; i < _block_param->solver_size(); ++i) {
		if (_block_param->solver(i).name() == name) {			
			return std::shared_ptr<deepflow::SolverParam>(_block_param->mutable_solver(i));
		}
	}
	return 0;
}

std::shared_ptr<deepflow::NodeParam> Block::find_node_by_output__name(const std::string & output_name) const
{
	for (int i = 0; i < _block_param->node_size(); ++i) {	
		for (auto output : _block_param->node(i).output())
			if (output == output_name) {
				return std::shared_ptr<deepflow::NodeParam>(_block_param->mutable_node(i));
			}
	}
	return 0;
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
