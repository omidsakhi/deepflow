#pragma once

#include "core/node.h"

class DeepFlowDllExport Block : public Node {

public: // Node	
	int minNumInputs();
	int minNumOutputs();
	void initForward();
	void initBackward();
	void forward();
	void backward();
	std::string to_cpp() const;
	ForwardType forwardType() { return DEPENDS_ON_OUTPUTS; }
	BackwardType backwardType() { return DEPENDS_ON_INPUTS; }

public: // Block
	Block();
	Block(deepflow::NodeParam &param);
	deepflow::NodeParam* find_node_by_name(const std::string &name) const;
	deepflow::SolverParam* find_solver_by_name(const std::string &name) const;
	deepflow::InitParam* find_initializer_by_name(const std::string &name) const;
	deepflow::NodeParam* find_node_by_output__name(const std::string & output_name) const;
	std::list<deepflow::NodeParam*> find_nodes_by_input_name(const std::string & input_name) const;
	std::string get_unique_node_name(const std::string &prefix) const;
	std::string get_unique_solver_name(const std::string &prefix) const;
	std::string get_unique_initializer_name(const std::string &prefix) const;
	void save_as_binary(std::string file_path);
	void save_as_text(std::string file_path);
	deepflow::NodeParam* add_node();
	deepflow::PhaseParam* add_phase();
	deepflow::SolverParam* add_solver();
	deepflow::InitParam* add_initializer();
	void print_nodes();
	void print_phases();
	void load_from_binary(std::string file_path);
	google::protobuf::RepeatedPtrField<deepflow::PhaseParam> phases();
	google::protobuf::RepeatedPtrField<deepflow::NodeParam> nodes();
	google::protobuf::RepeatedPtrField<deepflow::SolverParam> solvers();
	google::protobuf::RepeatedPtrField<deepflow::InitParam> initializers();

private:
	deepflow::BlockParam *_block_param;

};
