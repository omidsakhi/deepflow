#pragma once

#include "core/export.h"

#include "core/deep_flow.h"
#include "nodes/place_holder.h"
#include "nodes/switch.h"
#include "nodes/multiplexer.h"
#include "nodes/add.h"
#include "nodes/loss.h"
#include "nodes/gaussian_kernel.h"

class Solver;
class Variable;
class Loss;

class DeepFlowDllExport Session {
	friend class DeepFlow;
	friend class Node;
public:
	Session() {}
	Session(std::shared_ptr<Block> block) { _block = block; }	
	void initialize(std::shared_ptr<ExecutionContext> execution_context = nullptr);
	void set_execution_context(std::shared_ptr<ExecutionContext> execution_context);
	void run(std::string phase, int max_epoch, int max_iter, bool print_iteration, bool print_epoch, int debug_level);
	void mem_usage(size_t *free_byte, size_t *total_byte, float *used_byte_percentage);
	std::string to_cpp() const;
	std::shared_ptr<PlaceHolder> get_placeholder(std::string name);	
	std::shared_ptr<Node> get_node(std::string name);
	std::list<std::shared_ptr<Node>> forward_path();
	void forward();
	void reset_gradients();
	void clamp(float min, float max);
	void backward();	
	void apply_solvers(std::initializer_list<std::string> solver_names = {});
	void set_enabled_solvers(bool state, std::initializer_list<std::string> solver_names = {});
	void set_learning_rate(float lr, std::initializer_list<std::string> solver_names = {});
	void save(std::string file_path, bool as_text = false);
	void print_total_parameters();
	void print_variables_info();
	void print_nodes_info();
	std::shared_ptr<Node> end_node();
private:
	template <class T>
	std::list<std::shared_ptr<T>> _get_nodes(std::string execution_phase);
	bool is_end_node(std::shared_ptr<Node> node) const;
	bool is_head_node(std::shared_ptr<Node> node) const;
	std::list<std::shared_ptr<Node>> _get_all_end_nodes(std::string execution_phase) const;
	std::list<std::shared_ptr<Node>> _get_all_head_nodes(std::string execution_phase) const;
	std::shared_ptr<Node> _find_node_by_name(const std::string &name) const;
	std::shared_ptr<NodeOutput> _find_node_output_by_name(const std::string &name) const;
	std::shared_ptr<Node> _create_node(deepflow::NodeParam *);
	std::shared_ptr<Solver> _create_solver(deepflow::SolverParam *);
	//void _execute_one_pass(std::shared_ptr<ExecutionContext> context, int *iteration, std::list<std::shared_ptr<Node>> *nodes, std::list<std::shared_ptr<Node>> *generators, std::list<std::shared_ptr<Node>> *end_nodes, std::list<std::shared_ptr<Node>> *head_nodes, std::list<std::shared_ptr<Variable>> *variable_nodes, int max_iter, bool print_iteration);
	void _insert_splits();
private:
	bool _initialized = false;
	std::shared_ptr<ExecutionContext> _execution_context;
	std::shared_ptr<Block> _block;
	std::list<std::shared_ptr<Node>> _nodes;
	std::list<std::shared_ptr<Variable>> _variables;
	std::map<std::string, deepflow::PhaseParam_PhaseBehaviour> _phases;
	std::map<std::shared_ptr<Variable>, std::shared_ptr<Solver>> _solvers;
};

template<class T>
inline std::list<std::shared_ptr<T>> Session::_get_nodes(std::string execution_phase)
{
	std::list<std::shared_ptr<T>> list;
	if (execution_phase.empty()) {
		for (auto node : _nodes) {
			auto node_after_cast = std::dynamic_pointer_cast<T>(node);
			if (node_after_cast)
				list.push_back(node_after_cast);
		}
	}
	else {
		for (auto node : _nodes) {
			if (node->includePhase(execution_phase)) {
				auto node_after_cast = std::dynamic_pointer_cast<T>(node);
				if (node_after_cast)
					list.push_back(node_after_cast);
			}
		}
	}
	return list;
}
