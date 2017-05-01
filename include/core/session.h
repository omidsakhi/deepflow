#pragma once

#include "core/export.h"

#include "core/deep_flow.h"

class Solver;
class Variable;
class Loss;

class DeepFlowDllExport Session {
	friend class DeepFlow;
public:
	void setGraph(std::shared_ptr<GraphParam> graph);
	void initialize();
	void run(std::string phase, int max_epoch, int max_iter, bool print_iteration, bool print_epoch, int debug_level);
	std::string to_cpp() const;
	//void save_as_binary(std::string filePath, bool include_inits);
	//void save_as_text(std::string filePath, bool include_weights = false, bool include_inits = false);
private:
	template <class T>
	std::list<std::shared_ptr<T>> _get_nodes(std::string execution_phase);
	std::list<std::shared_ptr<Node>> _get_end_nodes(std::string execution_phase) const;
	std::shared_ptr<Node> _find_node_by_name(const std::string &name) const;
	std::shared_ptr<NodeOutput> _find_node_output_by_name(const std::string &name) const;
	std::shared_ptr<Node> _create_node(const NodeParam &);
	std::shared_ptr<Solver> _create_solver(const SolverParam &);
	void _execute_one_pass(std::shared_ptr<ExecutionContext> context, int *iteration, std::list<std::shared_ptr<Node>> *nodes, std::list<std::shared_ptr<Generator>> *generators, std::list<std::shared_ptr<Node>> *end_nodes, std::list<std::shared_ptr<Variable>> *variable_nodes, int max_iter, bool train, bool print_iteration);
private:
	bool _initialized = false;
	std::shared_ptr<GraphParam> _graph;
	std::list<std::shared_ptr<Node>> _nodes;	
	std::list<std::shared_ptr<Variable>> _variables;
	std::map<std::string, PhaseParam_PhaseBehaviour> _phases;
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
