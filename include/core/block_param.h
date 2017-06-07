#pragma once

#include "proto/deepflow.pb.h"
#include "core/export.h"

#include <list>

class DeepFlowDllExport BlockParam {
public:
	friend class Block;
	
	BlockParam() {}	

	deepflow::NodeParam* find_node_param_by_name(const std::string &node_name);
	int find_node_param_index_by_name(const std::string & output_name);
	deepflow::SolverParam* find_solver_param_by_name(const std::string &name);
	deepflow::InitParam* find_initializer_param_by_name(const std::string &name);
	deepflow::NodeParam* find_node_param_by_output_name(const std::string & output_name);
	std::list<deepflow::NodeParam*> find_node_params_by_input_name(const std::string & input_name);
	std::string get_unique_node_param_name(const std::string &prefix);
	std::string get_unique_solver_param_name(const std::string &prefix);
	std::string get_unique_initializer_param_name(const std::string &prefix);
	void replace_node_params_input(const std::string &search, const std::string &replace, const std::initializer_list<deepflow::NodeParam *> exclude_nodes);
	void remove_and_reconnect_node_param_by_name(const std::string &node_name);
	void remove_single_node_param(const std::string &node_name);
	void remove_node_params(std::initializer_list<std::string> node_names);
	void set_phase_for_node_params(std::string phase, std::initializer_list<std::string> node_names);
	void set_solver_for_variable_params(std::string solver, std::initializer_list<std::string> variable_names);
	void save_as_binary(std::string file_path);
	void save_as_text(std::string file_path);
	deepflow::NodeParam* add_node_param();
	deepflow::PhaseParam* add_phase_param();
	deepflow::SolverParam* add_solver_param();
	deepflow::InitParam* add_initializer_param();
	void print_node_params();
	void print_phase_params();
	void load_from_binary(std::string file_path);
	google::protobuf::RepeatedPtrField<deepflow::PhaseParam> phase_params();
	google::protobuf::RepeatedPtrField<deepflow::NodeParam> node_params();
	google::protobuf::RepeatedPtrField<deepflow::SolverParam> solver_params();
	google::protobuf::RepeatedPtrField<deepflow::InitParam> initializer_params();
	deepflow::BlockParam & block_param() { return _block_param; }
protected:
	std::string _name;
	deepflow::BlockParam _block_param;
};
