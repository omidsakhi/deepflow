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
	Block(deepflow::NodeParam &param);
	std::shared_ptr<deepflow::NodeParam> find_node_by_name(const std::string &name) const;
	std::shared_ptr<deepflow::SolverParam> find_solver_by_name(const std::string &name) const;
	std::shared_ptr<deepflow::NodeParam> find_node_by_output__name(const std::string & output_name) const;
	std::string get_unique_node_name(const std::string &prefix) const;
	std::string get_unique_solver_name(const std::string &prefix) const;
	void save_as_binary(std::string file_path);
	void save_as_text(std::string file_path);
	deepflow::NodeParam* add_node();
	deepflow::PhaseParam* add_phase();
	deepflow::SolverParam* add_solver();
	void print_nodes();
	void load_from_binary(std::string file_path);

private:	
	deepflow::BlockParam *param;
};