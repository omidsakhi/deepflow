#pragma once

#include "core/export.h"

#include "core/deep_flow.h"
#include "nodes/place_holder.h"
#include "nodes/switch.h"
#include "nodes/multiplexer.h"
#include "nodes/add.h"
#include "nodes/loss.h"
#include "nodes/gaussian_kernel.h"
#include "nodes/psnr.h"
#include "nodes/accumulator.h"
#include "nodes/print.h"
#include "nodes/logger.h"
#include "nodes/image_writer.h"

class Solver;
class Variable;
class Loss;

class DeepFlowDllExport Session {
	friend class DeepFlow;
	friend class Node;
public:
	Session() {}
	Session(std::shared_ptr<Block> block) { _block = block; }
	void create_nodes();
	void initialize(std::shared_ptr<ExecutionContext> execution_context = nullptr);
	void set_execution_context(std::shared_ptr<ExecutionContext> execution_context);	
	void mem_usage(size_t *free_byte, size_t *total_byte, float *used_byte_percentage);
	std::string to_cpp(const std::string &scope = "") const;
	std::shared_ptr<PlaceHolder> get_placeholder(const std::string &name, const std::string &scope = "");
	template <class T = Node>
	std::shared_ptr<T> get_node(std::string name, std::string scope = "", bool validate = true);	
	void forward(std::list<std::shared_ptr<Node>> end_nodes, std::list<std::pair<std::shared_ptr<PlaceHolder>, std::shared_ptr<Tensor>>> feed_list = {});
	void forward(const std::string &scope, std::list<std::pair<std::shared_ptr<PlaceHolder>, std::shared_ptr<Tensor>>> feed_list = {});
	void reset_gradients(const std::string &scope);
	void clamp(float min, float max, const std::string &scope);
	void backward(std::list<std::shared_ptr<Node>> end_nodes, std::list<std::pair<std::shared_ptr<Node>, std::shared_ptr<Tensor>>> feed_list = {});
	void backward(const std::string &scope, std::list<std::pair<std::shared_ptr<Node>, std::shared_ptr<Tensor>>> feed_list = {});
	void apply_solvers(std::list<std::string> solver_names = {});
	void apply_solvers(const std::string &scope);
	void set_enabled_solvers(bool state, std::list<std::string> solver_names = {});
	void set_enabled_solvers(bool state, const std::string &scope);
	void set_learning_rate(float lr, std::list<std::string> solver_names = {});
	void set_learning_rate(float lr, const std::string &scope);
	void save(std::string file_path, bool as_text = false);
	void print_total_parameters(const std::string &scope);
	void print_variables_info(const std::string &scope);
	void print_nodes_info(const std::string &scope);
	void print_nodes(const std::string &scope);
	std::shared_ptr<Node> end_node(const std::string &scope) const;
	std::list<std::shared_ptr<Node>> end_nodes(const std::string &scope) const;
	bool check_quit() { return _execution_context->quit; }
private:
	template <class T>
	std::list<std::shared_ptr<T>> _get_nodes(const std::string &scope);
	bool is_end_node(std::shared_ptr<Node> node) const;
	bool is_head_node(std::shared_ptr<Node> node) const;	
	std::shared_ptr<Node> _find_node_by_name(const std::string &name, const std::string &scope) const;
	std::shared_ptr<NodeOutput> _find_node_output_by_name(const std::string &name) const;
	std::shared_ptr<Node> _create_node(deepflow::NodeParam *);
	std::shared_ptr<Solver> _create_solver(deepflow::SolverParam *);	
	void _insert_splits();
private:
	bool _created = false;
	bool _initialized = false;
	std::shared_ptr<ExecutionContext> _execution_context;
	std::shared_ptr<Block> _block;
	std::list<std::shared_ptr<Node>> _nodes;
	std::list<std::shared_ptr<Variable>> _variables;	
	std::map<std::shared_ptr<Variable>, std::shared_ptr<Solver>> _solvers;
};

template<class T>
inline std::list<std::shared_ptr<T>> Session::_get_nodes(const std::string &scope)
{
	std::list<std::shared_ptr<T>> list;
	if (scope.empty()) {
		for (auto node : _nodes) {
			auto node_after_cast = std::dynamic_pointer_cast<T>(node);
			if (node_after_cast)
				list.push_back(node_after_cast);
		}
	}
	else {
		for (auto node : _nodes) {
			if (node->scope() == scope) {
				auto node_after_cast = std::dynamic_pointer_cast<T>(node);
				if (node_after_cast)
					list.push_back(node_after_cast);
			}
		}
	}
	return list;
}

template <class T>
std::shared_ptr<T> Session::get_node(std::string name, std::string scope, bool validate)
{
	auto node = _find_node_by_name(name, scope);
	auto ref = std::dynamic_pointer_cast<T>(node);
	if (ref == 0 && validate) {
		std::cout << "[FAILED] - " << name << " node does not exist in the graph.";
		exit(-1);
	}
	return ref;
}

