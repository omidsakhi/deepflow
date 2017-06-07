#include "core/block_param.h"

#include <google/protobuf/text_format.h>

#include <glog/logging.h>

#include <fstream>

deepflow::NodeParam* BlockParam::find_node_param_by_name(const std::string & name)
{
	for (int i = 0; i < _block_param.node_size(); ++i) {
		if (_block_param.node(i).name() == name) {
			return _block_param.mutable_node(i);
		}
	}
	return 0;
}

int BlockParam::find_node_param_index_by_name(const std::string & name)
{
	for (int i = 0; i < _block_param.node_size(); ++i) {
		if (_block_param.node(i).name() == name) {
			return i;
		}
	}
	return -1;
}

deepflow::SolverParam* BlockParam::find_solver_param_by_name(const std::string & name)
{
	for (int i = 0; i < _block_param.solver_size(); ++i) {
		if (_block_param.solver(i).name() == name) {
			return _block_param.mutable_solver(i);
		}
	}
	return 0;
}

deepflow::InitParam * BlockParam::find_initializer_param_by_name(const std::string & name)
{
	for (int i = 0; i < _block_param.initializer_size(); ++i) {
		if (_block_param.initializer(i).name() == name) {
			return _block_param.mutable_initializer(i);
		}
	}
	return 0;
}

deepflow::NodeParam* BlockParam::find_node_param_by_output_name(const std::string & output_name)
{
	for (int i = 0; i < _block_param.node_size(); ++i) {
		for (auto output : _block_param.node(i).output())
			if (output == output_name) {
				return _block_param.mutable_node(i);
			}
	}
	return 0;
}

std::list<deepflow::NodeParam*> BlockParam::find_node_params_by_input_name(const std::string & input_name)
{
	auto list = std::list<deepflow::NodeParam*>();
	for (int i = 0; i < _block_param.node_size(); ++i) {
		for (auto input : _block_param.node(i).input())
			if (input == input_name) {
				list.push_back(_block_param.mutable_node(i));
			}
	}
	return list;
}

std::string BlockParam::get_unique_node_param_name(const std::string & prefix)
{
	if (find_node_param_by_name(prefix) == 0)
		return prefix;
	int index = 1;
	std::string nodeName = prefix + "_" + std::to_string(index);
	while (find_node_param_by_name(nodeName) != 0) {
		index++;
		nodeName = prefix + "_" + std::to_string(index);
	}
	return nodeName;
}

std::string BlockParam::get_unique_solver_param_name(const std::string & prefix)
{
	if (find_solver_param_by_name(prefix) == 0)
		return prefix;
	int index = 1;
	std::string nodeName = prefix + "_" + std::to_string(index);
	while (find_solver_param_by_name(nodeName) != 0) {
		index++;
		nodeName = prefix + "_" + std::to_string(index);
	}
	return nodeName;
}

std::string BlockParam::get_unique_initializer_param_name(const std::string & prefix)
{
	if (find_initializer_param_by_name(prefix) == 0)
		return prefix;
	int index = 1;
	std::string initName = prefix + "_" + std::to_string(index);
	while (find_initializer_param_by_name(initName) != 0) {
		index++;
		initName = prefix + "_" + std::to_string(index);
	}
	return initName;
}

void BlockParam::replace_node_params_input(const std::string & search, const std::string & replace, const std::initializer_list<deepflow::NodeParam *> exclude_nodes)
{
	auto affected_input_nodes = find_node_params_by_input_name(search);
	for (auto node : exclude_nodes)
		affected_input_nodes.remove(node);
	for (auto affected_input_node : affected_input_nodes)
		for (int i = 0; i < affected_input_node->input_size(); i++) {
			if (affected_input_node->input(i) == search) {
				affected_input_node->set_input(i, replace);
			}
		}
}

void BlockParam::remove_and_reconnect_node_param_by_name(const std::string & node_name)
{
	deepflow::NodeParam *node = find_node_param_by_name(node_name);
	LOG_IF(FATAL, node == nullptr) << "Failed to find node with name " << node_name;
	int num_inputs = node->input_size();
	int num_outputs = node->output_size();
	LOG_IF(FATAL, num_outputs > num_inputs) << "Failed to remove node " << node_name << " due to num_outputs > num_inputs";
	for (int i = 0; i < num_outputs; i++) {
		replace_node_params_input(node->output(i), node->input(i), { node });
	}
	remove_single_node_param(node_name);
}

void BlockParam::remove_single_node_param(const std::string & node_name)
{
	for (auto itr = _block_param.node().begin(); itr != _block_param.node().end(); itr++) {
		if (itr->name() == node_name) {
			_block_param.mutable_node()->erase(itr);
			break;
		}
	}
}

void BlockParam::remove_node_params(std::initializer_list<std::string> node_names)
{
	std::list<deepflow::NodeParam*> queue;
	for (auto name : node_names) {
		deepflow::NodeParam *node = find_node_param_by_name(name);
		if (node != nullptr)
			queue.push_back(node);
	}
	std::list<deepflow::NodeParam*> nodes_to_remove;
	while (!queue.empty()) {
		deepflow::NodeParam *node = queue.front();
		queue.pop_front();
		nodes_to_remove.push_back(node);
		for (auto node_output : node->output()) {
			auto downstream_nodes = find_node_params_by_input_name(node_output);
			for (auto another_node : downstream_nodes) {
				queue.push_back(another_node);
			}
		}
	}
	while (!nodes_to_remove.empty()) {
		deepflow::NodeParam *node = nodes_to_remove.front();
		nodes_to_remove.pop_front();
		remove_single_node_param(node->name());
	}
}

void BlockParam::set_phase_for_node_params(std::string phase, std::initializer_list<std::string> node_names)
{
	if (node_names.size() == 0) {
		for (int i = 0; i < _block_param.node_size(); ++i)
			_block_param.mutable_node(i)->add_phase(phase);
	}
	else {
		for (auto name : node_names) {
			deepflow::NodeParam *node = find_node_param_by_name(name);
			LOG_IF(FATAL, node == nullptr) << "Failed to find node with name " << name;
			node->add_phase(phase);
		}
	}
}

void BlockParam::set_solver_for_variable_params(std::string solver, std::initializer_list<std::string> variable_names)
{
	if (variable_names.size() == 0) {
		for (int i = 0; i < _block_param.node_size(); ++i) {
			deepflow::NodeParam * node = _block_param.mutable_node(i);
			if (node->has_variable_param()) {
				node->mutable_variable_param()->set_solver_name(solver);
			}
		}
	}
	else {
		for (auto name : variable_names) {
			deepflow::NodeParam *node = find_node_param_by_name(name);
			LOG_IF(FATAL, node == nullptr) << "Failed to find node with name " << name;
			if (node->has_variable_param()) {
				node->mutable_variable_param()->set_solver_name(solver);
			}
			else {
				LOG(FATAL) << "Node " << name << " is not a variable.";
			}
		}
	}
}

void BlockParam::save_as_binary(std::string file_path)
{
	std::fstream output(file_path, std::ios::out | std::ios::trunc | std::ios::binary);
	LOG_IF(FATAL, !_block_param.SerializeToOstream(&output)) << "Failed to write block to " << file_path;
	output.close();
}

void BlockParam::save_as_text(std::string file_path)
{
	std::string text;
	google::protobuf::TextFormat::PrintToString(_block_param, &text);
	std::ofstream out(file_path);
	out << text;
	out.close();
}

deepflow::NodeParam* BlockParam::add_node_param()
{
	return _block_param.add_node();
}

deepflow::PhaseParam * BlockParam::add_phase_param()
{
	return _block_param.add_phase();
}

deepflow::SolverParam * BlockParam::add_solver_param()
{
	return _block_param.add_solver();
}

deepflow::InitParam * BlockParam::add_initializer_param()
{
	return _block_param.add_initializer();
}

void BlockParam::print_node_params()
{
	for (auto node : _block_param.node()) {
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

void BlockParam::print_phase_params()
{
	for (auto phase : _block_param.phase()) {
		LOG(INFO) << phase.phase() << " <-> " << deepflow::PhaseParam_PhaseBehaviour_Name(phase.behaviour());
	}
}

void BlockParam::load_from_binary(std::string file_path)
{
	std::fstream input(file_path, std::ios::in | std::ios::binary);
	LOG_IF(FATAL, !_block_param.ParseFromIstream(&input)) << "Failed to read binary block from " << file_path;
	input.close();
}

google::protobuf::RepeatedPtrField<deepflow::PhaseParam> BlockParam::phase_params()
{
	return _block_param.phase();
}

google::protobuf::RepeatedPtrField<deepflow::NodeParam> BlockParam::node_params()
{
	return _block_param.node();
}

google::protobuf::RepeatedPtrField<deepflow::SolverParam> BlockParam::solver_params()
{
	return _block_param.solver();
}

google::protobuf::RepeatedPtrField<deepflow::InitParam> BlockParam::initializer_params()
{
	return _block_param.initializer();
}