#include "core/session.h"

#include "nodes/variable.h"
#include "nodes/place_holder.h"
#include "nodes/phaseplexer.h"

#include "initializers/fill.h"
#include "initializers/index_fill.h"
#include "initializers/random_uniform.h"
#include "initializers/random_normal.h"
#include "initializers/step.h"
#include "initializers/three_state.h"

#include "core/solver.h"

#include "solvers/gain_solver.h"
#include "solvers/sgd_solver.h"
#include "solvers/adam_solver.h"
#include "solvers/adadelta_solver.h"

#include "nodes/add.h"
#include "nodes/matmul.h"
#include "nodes/leaky_relu.h"
#include "nodes/softmax.h"
#include "nodes/exp.h"
#include "nodes/abs.h"
#include "nodes/square.h"
#include "nodes/bias_add.h"
#include "nodes/dropout.h"
#include "nodes/convolution_2d.h"
#include "nodes/pooling.h"
#include "nodes/reduce.h"
#include "nodes/equal.h"
#include "nodes/cast_float.h"
#include "nodes/display.h"
#include "nodes/transposed_conv_2d.h"
#include "nodes/square_error.h"
#include "nodes/activation.h"
#include "nodes/psnr.h"
#include "nodes/random_selector.h"
#include "nodes/block.h"
#include "nodes/restructure.h"
#include "nodes/accumulator.h"
#include "nodes/print.h"
#include "nodes/psnr.h"
#include "nodes/logger.h"
#include "nodes/image_reader.h"
#include "nodes/multiplexer.h"
#include "nodes/batch_normalization.h"
#include "nodes/dot.h"
#include "nodes/replay_memory.h"
#include "nodes/sio_output.h"
#include "nodes/loss.h"
#include "nodes/log.h"
#include "nodes/lifting.h"
#include "nodes/patching.h"
#include "nodes/reduce_all.h"
#include "nodes/upsample.h"
#include "nodes/image_writer.h"
#include "nodes/resize.h"
#include "nodes/split.h"
#include "nodes/switch.h"
#include "nodes/lrn.h"
#include "nodes/prelu.h"
#include "nodes/concate.h"
#include "nodes/reshape.h"

#include "generators/data_generator.h"
#include "generators/image_batch_reader.h"

#include <unordered_map>
#include <map>

#include<memory>

#include <vector>

#include <fstream>

#include <chrono>

#include <ctime>

std::shared_ptr<Initializer> _create_initializer(deepflow::InitParam *init_param) {
	
	if (init_param->has_fill_param()) {
		return std::make_shared<Fill>(init_param);
	}
	else if (init_param->has_index_fill_param()) {
		return std::make_shared<IndexFill>(init_param);
	}
	else if (init_param->has_random_uniform_param()) {
		return std::make_shared<RandomUniform>(init_param);
	}
	else if (init_param->has_step_param()) {
		return std::make_shared<Step>(init_param);
	}
	else if (init_param->has_random_normal_param()) {
		return std::make_shared<RandomNormal>(init_param);
	}
	else if (init_param->has_three_state_param()) {
		return std::make_shared<ThreeState>(init_param);
	}
	else {
		LOG(FATAL) << "Unsupported Initializer";
	}

	return NULL;
}

std::shared_ptr<Node> Session::_create_node(deepflow::NodeParam *node_param) {

	if (node_param->has_batch_normalization_param())
		return std::make_shared<BatchNormalization>(node_param);
	else if (node_param->has_image_batch_reader_param())
		return std::make_shared<ImageBatchReader>(node_param);
	else if (node_param->has_mnist_param())
		return std::make_shared<MNISTReader>(node_param);
	else if (node_param->has_data_generator_param()) {
		auto init_param = node_param->mutable_variable_param()->mutable_init_param();
		std::shared_ptr<Initializer> initializer = _create_initializer(init_param);
		return std::make_shared<DataGenerator>(initializer, node_param);
	}
	else if (node_param->has_variable_param()) {
		auto init_param = node_param->mutable_variable_param()->mutable_init_param();
		std::shared_ptr<Initializer> initializer = _create_initializer(init_param);
		return std::make_shared<Variable>(initializer, node_param);
	}
	else if (node_param->has_image_reader_param())
		return std::make_shared<ImageReader>(node_param);
	else if (node_param->has_transposed_conv_2d_param())
		return std::make_shared<TransposedConvolution2D>(node_param);
	else if (node_param->has_matmul_param())
		return std::make_shared<MatMul>(node_param);
	else if (node_param->has_conv_2d_param())
		return std::make_shared<Convolution2D>(node_param);
	else if (node_param->has_abs_param())
		return std::make_shared<Abs>(node_param);
	else if (node_param->has_log_param())
		return std::make_shared<Log>(node_param);
	else if (node_param->has_loss_param())
		return std::make_shared<Loss>(node_param);
	else if (node_param->has_reduce_all_param())
		return std::make_shared<ReduceAll>(node_param);
	else if (node_param->has_square_error_param())
		return std::make_shared<SquareError>(node_param);
	else if (node_param->has_patching_param())
		return std::make_shared<Patching>(node_param);
	else if (node_param->has_lifting_param())
		return std::make_shared<Lifting>(node_param);
	else if (node_param->has_resize_param())
		return std::make_shared<Resize>(node_param);
	else if (node_param->has_pooling_param())
		return std::make_shared<Pooling>(node_param);
	else if (node_param->has_upsample_param())
		return std::make_shared<Upsample>(node_param);
	else if (node_param->has_reduce_param())
		return std::make_shared<Reduce>(node_param);
	else if (node_param->has_display_param())
		return std::make_shared<Display>(node_param);
	else if (node_param->has_leaky_relu_param())
		return std::make_shared<LeakyRelu>(node_param);
	else if (node_param->has_prelu_param())
		return std::make_shared<PRelu>(node_param);
	else if (node_param->has_softmax_param())
		return std::make_shared<Softmax>(node_param);
	else if (node_param->has_dropout_param())
		return std::make_shared<Dropout>(node_param);
	else if (node_param->has_image_writer_param())
		return std::make_shared<ImageWriter>(node_param);
	else if (node_param->has_replay_memory_param())
		return std::make_shared<ReplayMemory>(node_param);
	else if (node_param->has_add_param())
		return std::make_shared<Add>(node_param);
	else if (node_param->has_exp_param())
		return std::make_shared<Exp>(node_param);
	else if (node_param->has_bias_add_param())
		return std::make_shared<BiasAdd>(node_param);
	else if (node_param->has_cast_float_param())
		return std::make_shared<CastFloat>(node_param);
	else if (node_param->has_equal_param())
		return std::make_shared<Equal>(node_param);
	else if (node_param->has_activation_param())
		return std::make_shared<Activation>(node_param);
	else if (node_param->has_concate_param())
		return std::make_shared<Concate>(node_param);
	else if (node_param->has_phaseplexer_param())
		return std::make_shared<Phaseplexer>(node_param);
	else if (node_param->has_place_holder_param())
		return std::make_shared<PlaceHolder>(node_param);
	else if (node_param->has_print_param())
		return std::make_shared<Print>(node_param);
	else if (node_param->has_logger_param())
		return std::make_shared<Logger>(node_param);
	else if (node_param->has_square_param())
		return std::make_shared<Square>(node_param);
	else if (node_param->has_restructure_param())
		return std::make_shared<Restructure>(node_param);
	else if (node_param->has_psnr_param())
		return std::make_shared<Psnr>(node_param);
	else if (node_param->has_random_selector_param())
		return std::make_shared<RandomSelector>(node_param);
	else if (node_param->has_multiplexer_param())
		return std::make_shared<Multiplexer>(node_param);
	else if (node_param->has_accumulator_param())
		return std::make_shared<Accumulator>(node_param);
	else if (node_param->has_dot_param())
		return std::make_shared<Dot>(node_param);
	else if (node_param->has_sio_output_param())
		return std::make_shared<SIOOutput>(node_param);
	else if (node_param->has_split_param())
		return std::make_shared<Split>(node_param);
	else if (node_param->has_switch_param())
		return std::make_shared<Switch>(node_param);
	else if (node_param->has_reshape_param())
		return std::make_shared<Reshape>(node_param);
	else if (node_param->has_lrn_param())
		return std::make_shared<LRN>(node_param);
	else {
		LOG(FATAL) << "Unsupported Node";
	}

	return 0;
}

std::shared_ptr<Solver> Session::_create_solver(deepflow::SolverParam *solver_param) {
	if (solver_param->has_gain_solver()) {
		return std::make_shared<GainSolver>(solver_param);
	}
	else if (solver_param->has_sgd_solver()) {
		return std::make_shared<SGDSolver>(solver_param);
	}
	else if (solver_param->has_adam_solver()) {
		return std::make_shared<AdamSolver>(solver_param);
	}
	else if (solver_param->has_adadelta_solver()) {
		return std::make_shared<AdaDeltaSolver>(solver_param);
	}
	else {
		LOG(FATAL) << "Unsupported Solver";
	}

	return 0;
}

void Session::initialize(std::shared_ptr<ExecutionContext> execution_context) {
	if (_initialized == true)
		return;
	
	_initialized = true;	

	for (auto phase_param : _block->phase_params())
		_phases.insert(std::pair<std::string, deepflow::PhaseParam_PhaseBehaviour>(phase_param.phase(), phase_param.behaviour()));
	
	// creating nodes
	for (int i=0; i < _block->block_param()->node_size(); ++i)
	{
		auto node_param = _block->block_param()->mutable_node(i);
		auto node = _create_node(node_param);
		node->createIO();
		LOG_IF(FATAL, node->inputs().size() != node->param()->input_size()) << "Node " << node->name() << "'s input size " << node->inputs().size() << " does not match the one specified in proto (" << node->param()->input_size() << ")";		
		_nodes.push_back(node);
	}

	// connecting node inputs/outputs
	for (auto node : _nodes) {
		for (int i = 0; i < node->param()->input_size(); ++i) {			
			const std::string terminal_name = node->param()->input(i);
			if (!terminal_name.empty()) {
				auto terminal = _find_node_output_by_name(terminal_name);
				LOG_IF(FATAL, terminal == 0) << "Failed to find " << terminal_name << " for node " << node->name();
				node->input(i)->connect(terminal);
			}
		}
	}
	
	// caching the name of all variables
	_variables = _get_nodes<Variable>("");

	for (auto var : _variables) {
		std::shared_ptr<Solver> solver;
		std::string var_solver_name = var->param()->variable_param().solver_name();
		for (int i=0; i < _block->block_param()->solver_size(); ++i) 
		{
			auto solver_param = _block->block_param()->mutable_solver(i);
			if (var_solver_name == solver_param->name()) {
				solver = _create_solver(solver_param);
				break;
			}
		}
		if (!var_solver_name.empty() && solver == nullptr) {
			LOG(FATAL) << "solver " << var_solver_name << " couldn't be found for variable " << var->name();
		}
		else if (solver) {
			_solvers.insert(std::pair<std::shared_ptr<Variable>, std::shared_ptr<Solver>>(var, solver));
			LOG(INFO) << "variable " << var->name() << " <-> Solver " << solver->name();
			auto enable_input = solver->param()->enable_input();
			if (!enable_input.empty()) {
				auto terminal = _find_node_output_by_name(enable_input);
				LOG_IF(FATAL, terminal == 0) << "Failed to find " << enable_input << " as the input for solver " << solver->param()->name() << " in variable " << var_solver_name;
				solver->create_enable_input();
				solver->enable_input()->connect(terminal);
				LOG(INFO) << "solver " << solver->name() << " Enable <-> Node " << terminal->parentNode()->name();
			}
		}
		else {
			LOG(INFO) << "variable " << var->name() << " <-> Constant";
		}
	}

	_insert_splits();

	size_t free_byte_before, free_byte_after;
	size_t total_byte;	
	std::srand(std::time(0));
	std::list<std::shared_ptr<Node>> queue = _nodes;
	while (!queue.empty()) {
		auto node = queue.front();
		queue.pop_front();
		if (node->isInitialized())
			continue;
		bool resolved = true;
		for (auto input : node->inputs()) {
			auto connectedNode = input->connectedNode();
			if (connectedNode) {
				if (connectedNode->isInitialized() == false) {
					resolved = false;
					break;
				}
			}
		}
		if (resolved) {
			mem_usage(&free_byte_before, &total_byte, 0);
			node->init();						
			mem_usage(&free_byte_after, &total_byte, 0);
			std::string shape;
			for (auto output : node->outputs())
				shape += output->value()->shape() + " , ";			
			LOG(INFO) << node->op_name() << " " << node->name() << " | " << (10e-6f * (free_byte_before - free_byte_after)) << "MB | " << shape;
			node->setInitialized(true);
		}
		else {
			queue.push_back(node);
		}
	}

	if (execution_context) {
		set_execution_context(execution_context);
	}
	else {
		auto execution_context = std::make_shared<ExecutionContext>();
		set_execution_context(execution_context);
	}
	
}

void Session::_insert_splits()
{
	int total_split_nodes = 0;

	for (auto node : _nodes) {
		for (auto output : node->outputs()) {			
			auto connected_terminals = output->connectedTerminals();
			LOG_IF(FATAL, connected_terminals.size() > 2) << "We don't support more than 2 connected nodes to one output at this time | " << node->name();
			if (connected_terminals.size() == 2) {

				auto node_param = deepflow::NodeParam();
				node_param.set_name("split_" + output->name());
				node_param.add_output(node_param.name() + "_output_0");
				node_param.add_output(node_param.name() + "_output_1");				
				node_param.add_input(node_param.name() + "_input");
				node_param.mutable_split_param();

				auto new_split_node = _create_node(&node_param);
				new_split_node->createIO();

				output->disconnect();
				new_split_node->input(0)->connect(output);				

				int i = 0;
				for (auto t : connected_terminals) {
					auto tcast = std::dynamic_pointer_cast<NodeInput>(t);
					if (tcast) {
						tcast->disconnect();
						tcast->connect(new_split_node->output(i++));
					}
				}

				_nodes.push_back(new_split_node);
				total_split_nodes++;
			}
		}
	}

	for (auto node : _nodes) {
		for (auto output : node->outputs()) {
			auto connected_terminals = output->connectedTerminals();
			LOG_IF(FATAL, connected_terminals.size() > 1);
		}
	}

	LOG(INFO) << total_split_nodes << " total split nodes.";
}

void Session::set_execution_context(std::shared_ptr<ExecutionContext> execution_context)
{
	_execution_context = execution_context;
	for (auto node : _nodes)
		node->setExecutionContext(execution_context);
}

std::shared_ptr<Node> Session::_find_node_by_name(const std::string &name) const {
	for (auto node : _nodes) {
		if (node->name() == name)
			return node;
	}
	return 0;
}

std::list<std::shared_ptr<Node>> Session::_get_all_end_nodes(std::string execution_phase) const
{
	std::list<std::shared_ptr<Node>> list;
	for (auto node : _nodes) {
		if (node->includePhase(execution_phase)) {
			int num_connected_inputs = 0;
			for (auto input : node->inputs())
				num_connected_inputs += input->connectedNode() ? 1 : 0;
			int num_connected_outputs = 0;
			for (auto output : node->outputs())
				num_connected_outputs += output->connectedNodes().size();			
			if (num_connected_outputs == 0 && num_connected_inputs != 0)
				list.push_back(node);
		}
	}
	return list;
}

std::list<std::shared_ptr<Node>> Session::_get_all_head_nodes(std::string execution_phase) const
{
	std::list<std::shared_ptr<Node>> list;
	for (auto node : _nodes) {
		if (node->includePhase(execution_phase)) {
			int num_connected_inputs = 0;
			for (auto input : node->inputs())
				num_connected_inputs += input->connectedNode() ? 1 : 0;
			int num_connected_outputs = 0;
			for (auto output : node->outputs())
				num_connected_outputs += output->connectedNodes().size();
			if (num_connected_inputs == 0 && num_connected_outputs != 0)
				list.push_back(node);
		}
	}
	return list;
}

/*
std::list<std::shared_ptr<Node>> Session::_get_forward_path(std::string execution_phase) const
{

	/*


	std::list<std::shared_ptr<Node>> end_nodes = _get_all_end_nodes("");
	std::list<std::list<std::shared_ptr<Node>>> paths;
	std::list<std::shared_ptr<Node>> abandon;
	for (auto node : end_nodes) {
		int visit_token = rand();
		for (auto node : abandon)
			node->setVisitToken(visit_token);
		std::list<std::shared_ptr<Node>> path;
		std::list<std::shared_ptr<Node>> list;		
		list.push_back(node);
		bool valid_path = true;
		while (valid_path && list.size() > 0) {
			auto node = list.front();
			list.pop_front();
			if (node->visitToken() == visit_token) {
				continue;
			}
			auto inputNodes = node->inputNodes();
			auto outputNodes = node->outputNodes();
			if (inputNodes.size() == 0 && outputNodes.size() == 0) {
				valid_path = false;				
			}
			if (valid_path && inputNodes.size() > 0) {
				bool found = false;
				for (auto inputNode : inputNodes) {
					for (auto outputNode : inputNode->outputNodes()) {
						if (outputNode == node) {
							found = true;
							break;
						}
					}
					if (found)
						break;
				}
				if (!found)	valid_path = false;
			}
			if (valid_path && outputNodes.size() > 0) {
				bool found = false;
				for (auto outputNode : outputNodes) {
					for (auto inputNode : outputNode->inputNodes()) {
						if (inputNode == node) {
							found = true;
							break;
						}
					}
					if (found) break;
				}
				if (!found) {
					valid_path = false;
				}
				else {
					bool all_visited = true;
					for (auto outputNode : outputNodes) {
						if (outputNode->visitToken() != visit_token) {
							bool valid = false;
							for (auto in : outputNode->inputNodes())
								if (in == node) {
									valid = true;
									break;
								}
							if (valid)
								all_visited = false;
							break;
						}
					}
					if (!all_visited) {
						list.push_back(node);
						continue;
					}
				}
			}
			node->setVisitToken(visit_token);
			if (valid_path) {
				path.push_front(node);				
				for (auto inputNode : node->inputNodes()) {
					list.push_back(inputNode);
				}
			}
			else {
				for (auto node : path)
					abandon.push_back(node);
				abandon.push_back(node);
			}
		}
		if (valid_path)
			paths.push_back(path);
	}

	std::list<std::shared_ptr<Node>> head_nodes = _get_all_head_nodes("");
	std::list<std::shared_ptr<Node>> valid_path;
	for (auto path : paths) {
		bool foundAHead = false;
		for (auto node : path) {
			for (auto head : head_nodes) {
				if (head == node) {
					foundAHead = true;
					break;
				}
			}
			if (foundAHead)
				break;
		}
		if (foundAHead) {
			for (auto node : path) {
				bool found = false;
				for (auto n2 : valid_path) {
					if (n2 == node) {
						found = true;
						break;
					}
				}
				if (!found)
					valid_path.push_back(node);
			}
		}
	}

	return valid_path;
	
}
*/

/*
std::list<std::shared_ptr<Node>> Session::_get_backward_path(std::string execution_phase) const
{

	std::list<std::shared_ptr<Node>> head_nodes = _get_all_head_nodes("");
	std::list<std::list<std::shared_ptr<Node>>> paths;	
	for (auto node : head_nodes) {		
		std::list<std::shared_ptr<Node>> path;
		std::list<std::shared_ptr<Node>> list;
		list.push_back(node);
		while (list.size() > 0) {
			auto node = list.front();
			int visit_token = rand();
			list.pop_front();
			if (node->visitToken() != visit_token) {
				path.push_front(node);
				node->setVisitToken(visit_token);
				for (auto outputNode : node->outputNodes()) {
					bool found = false;
					for (auto inputNode : outputNode->inputNodes()) {
						if (inputNode == node) {
							found = true;
							break;
						}
					}
					if (found)
						list.push_back(outputNode);
				}
			}
		}
		paths.push_back(path);
	}

	std::list<std::shared_ptr<Node>> end_nodes = _get_all_end_nodes("");
	std::list<std::shared_ptr<Node>> valid_path;
	for (auto path : paths) {
		bool foundEnd = false;
		for (auto node : path) {
			for (auto end : end_nodes) {
				if (end == node) {					
					foundEnd = true;
					break;
				}
			}
			if (foundEnd)
				break;
		}
		if (foundEnd) {
			for (auto node : path) {
				bool found = false;
				for (auto n2 : valid_path)
					if (n2 == node) {
						found = true;
						break;
					}
				if (!found)
				valid_path.push_back(node);
			}
		}
	}
	return valid_path;
}
*/

std::shared_ptr<NodeOutput> Session::_find_node_output_by_name(const std::string &name) const {
	for (auto node : _nodes) {
		for (auto terminal : node->outputs()) {
			if (terminal->name() == name)
				return terminal;
		}
	}
	return 0;
}

std::list<std::shared_ptr<Node>> Session::forward_path()
{
	for (auto node : _nodes) {
		for (auto t : node->outputs())
			t->setEnabled(false);
	}

	std::list<std::shared_ptr<Node>> nodes = _get_all_end_nodes("");
	std::list<std::shared_ptr<Node>> active_head_nodes;
	std::list<std::shared_ptr<Node>> visited_nodes;

	auto isVisited = [](std::list<std::shared_ptr<Node>> &list, std::shared_ptr<Node> node_to_check) {
		for (auto node : list)
			if (node == node_to_check)
				return true;
		return false;
	};

	int _verbose = 0;
	while (nodes.size() > 0) {
		auto node = nodes.front();
		_verbose = node->executionContext()->debug_level;
		nodes.pop_front();
		auto inputNodes = node->inputNodes();
		auto outputNodes = node->outputNodes();
		if (isVisited(visited_nodes, node)) {
			continue;
		}
		visited_nodes.push_back(node);
		if (inputNodes.size() == 0 && outputNodes.size() == 0) {
			continue;
		}
		if (inputNodes.size() > 0) {
			for (auto inputNode : inputNodes) {
				for (auto t : inputNode->outputs()) {
					for (auto c : t->connectedNodes()) {
						if (c == node) {
							t->setEnabled(true);
							break;
						}
					}
				}
				nodes.push_back(inputNode);
			}
		}
		else {
			active_head_nodes.push_back(node);
		}

	}


	nodes = active_head_nodes;
	std::list<std::shared_ptr<Node>> forward_path;
	visited_nodes.clear();

	while (nodes.size() > 0) {
		auto node = nodes.front();
		_verbose = node->executionContext()->debug_level;
		LOG_IF(INFO, _verbose > 3) << "FWRD POP " << node->name();
		nodes.pop_front();
		auto inputNodes = node->inputNodes();
		auto outputNodes = node->outputNodes();
		if (isVisited(visited_nodes, node)) {
			LOG_IF(INFO, _verbose > 3) << "END FOR " << node->name() << " | ALREADY VISITED.";
			continue;
		}
		if (inputNodes.size() == 0 && outputNodes.size() == 0) {
			visited_nodes.push_back(node);
			forward_path.push_front(node);
			LOG_IF(INFO, _verbose > 3) << "END FOR " << node->name() << "  | NO INPUT & OUTPUT.";
			continue;
		}
		if (inputNodes.size() > 0) {
			bool ready = true;
			for (auto inputNode : inputNodes) {
				if (!isVisited(visited_nodes, inputNode))
				{
					LOG_IF(INFO, _verbose > 3) << "REPUSH " << node->name() << " | " << inputNode->name();
					ready = false;
					break;
				}
			}
			if (!ready) {
				nodes.push_back(node);
				continue;
			}
		}
		visited_nodes.push_back(node);		
		forward_path.push_back(node);
		if (outputNodes.size() > 0) {
			for (auto outputT : node->outputs()) {
				if (outputT->enabled()) {
					for (auto outputNode : outputT->connectedNodes()) {
						LOG_IF(INFO, _verbose > 3) << "PUSH " << outputNode->name();
						nodes.push_back(outputNode);
					}
				}
			}
		}
	}
	return forward_path;	
}

void Session::forward()
{
	auto path = forward_path();
	while (path.size() > 0) {
		auto node = path.front();
		path.pop_front();
		int _verbose = node->executionContext()->debug_level;
		LOG_IF(INFO, _verbose > 2) << "FWRD -> " << node->name();
		node->forward();
	}
}

void Session::reset_gradients()
{
	for (auto var : _variables)
		var->reset_gradients();
}

void Session::clamp(float min, float max)
{
	for (auto var : _variables) {		
		var->clamp(min, max);
	}
}

void Session::backward()
{
	auto path = forward_path();
	while (path.size() > 0) {
		auto node = path.back();
		path.pop_back();
		int _verbose = node->executionContext()->debug_level;
		LOG_IF(INFO, _verbose > 2) << "BWRD -> " << node->name();
		node->backward();
	}
}

void Session::apply_solvers()
{
	std::list<std::shared_ptr<Variable>> variable_nodes = _get_nodes<Variable>("");
	for (auto var : variable_nodes) {
		auto map_var_to_solver = _solvers.find(var);
		if (map_var_to_solver != _solvers.end()) {
			map_var_to_solver->second->apply(var);
		}
	}
}

void Session::apply_solvers(std::initializer_list<std::string> solver_names)
{
	for (auto item : _solvers) {
		for (auto name : solver_names) {
			if (item.second->name() == name) {
				item.second->apply(item.first);				
			}
		}
	}
}

void Session::save(std::string file_path, bool as_text)
{
	for (auto node : _nodes)
		node->prep_for_saving();		
	if (as_text)
		_block->save_as_text(file_path);
	else
		_block->save_as_binary(file_path);
}

void Session::run(std::string phase, int max_epoch, int max_iter, bool print_iteration, bool print_epoch, int debug_level) {
	int iteration = 1;
	for (int epoch = 1; epoch <= max_epoch; ++epoch) {
		if (print_epoch)
			std::cout << "Epoch " << epoch << " -->" << std::endl;
		auto epoch_start = std::chrono::high_resolution_clock::now();
		forward();
		backward();
		apply_solvers();
		reset_gradients();
		auto epoch_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_epoch = epoch_end - epoch_start;
		if (print_epoch)
			std::cout << "<-- Epoch " << epoch << " Elapsed time: " << elapsed_epoch.count() << " seconds" << std::endl;
		if (_execution_context->quit == true)
			break;
		if (max_iter > 0 && iteration >= max_iter)
			break;
	}
}

/*
void Session::_execute_one_pass(std::shared_ptr<ExecutionContext> context, int *iteration, std::list<std::shared_ptr<Node>> *nodes, std::list<std::shared_ptr<Node>> *generators, std::list<std::shared_ptr<Node>> *end_nodes, std::list<std::shared_ptr<Node>> *head_nodes, std::list<std::shared_ptr<Variable>> *variable_nodes, int max_iter, bool print_iteration) {
	int iteration_per_epoch = 1;
	bool last_batch = false;	
	for (auto node : *nodes)
		node->setExecutionContext(context);
	bool train = _phases.find(context->phase)->second == deepflow::PhaseParam_PhaseBehaviour_TRAIN;
	bool validation = _phases.find(context->phase)->second == deepflow::PhaseParam_PhaseBehaviour_VALIDATION;
	if (train || validation) {
		int token = rand();
		for (auto node : *end_nodes) {
			node->_resolve_propagation(token);
		}
	}
	do {
		if (generators->size() > 0) {
			for (auto gen : *generators) {
				LOG_IF(FATAL, gen->isGenerator() == false) << "NOT A GENERATOR";
				if (gen->isLastBatch()) {
					last_batch = true;
					//LOG(INFO) << "LAST BATCH: " << std::dynamic_pointer_cast<Node>(gen)->name();
				}
			}
		}
		else {
			last_batch = true;
		}
		if (iteration)
			context->current_iteration = *iteration;
		context->current_iteration_per_epoch = iteration_per_epoch;
		context->last_batch = last_batch;
		if (iteration && print_iteration)
			std::cout << "  Iteration " << *iteration << std::endl;
		int token = rand();
		for (auto node : *end_nodes) {
			node->_forward(token);
		}
		if (train) {
			for (auto node : *variable_nodes)
				node->reset_gradients();
			int token = rand();
			for (auto node : *head_nodes) {
				node->_backward(token);
			}
			for (auto var : *variable_nodes) {
				auto map_var_to_solver = _solvers.find(var);
				if (map_var_to_solver != _solvers.end()) {										
					map_var_to_solver->second->apply(var);
				}
			}
		}
		if (iteration)
			(*iteration)++;
		iteration_per_epoch++;
		if (max_iter > 0 && iteration && (*iteration) >= max_iter)
			break;
	} while (last_batch == false && context->quit != true);
}

*/

/*

void Session::run(std::string phase, int max_epoch, int max_iter, bool print_iteration, bool print_epoch, int debug_level) {
	LOG(INFO) << "Executing graph for phase " << phase;
	LOG_IF(FATAL, _phases.find(phase) == _phases.end()) << "Specified phase " << phase << " is not defined.";	
	std::list<std::shared_ptr<Node>> execution_phase_generators;
	for (auto node : _nodes) {
		if (!phase.empty() && node->includePhase(phase) == false)
			continue;
		if (node->isGenerator())
			execution_phase_generators.push_back(node);
	}	
	std::list<std::shared_ptr<Node>> execution_end_nodes = _get_end_nodes(phase);
	std::list<std::shared_ptr<Node>> execution_head_nodes = _get_head_nodes(phase);
	std::string end_node_names;
	for (auto node : execution_end_nodes) {
		end_node_names += node->name() + " ";
	}
	LOG(INFO) << "End nodes: " << end_node_names;
	auto execution_context = std::make_shared<ExecutionContext>();
	deepflow::PhaseParam_PhaseBehaviour behaviour = _phases.find(phase)->second;
	execution_context->phase = phase;
	execution_context->phase_behaviour = behaviour;
	execution_context->debug_level = debug_level;
	if (behaviour == deepflow::PhaseParam_PhaseBehaviour_TRAIN) {
		std::list<std::shared_ptr<Variable>> execution_phase_variable_nodes = _get_nodes<Variable>(phase);
		std::string validation_phase;
		for (auto phase : _phases) {
			if (phase.second == deepflow::PhaseParam_PhaseBehaviour_VALIDATION) {
				validation_phase = phase.first;
				break;
			}
		}
		std::list<std::shared_ptr<Node>> validation_phase_generators;
		std::list<std::shared_ptr<Node>> validation_end_nodes, validation_head_nodes;
		std::shared_ptr<ExecutionContext> validation_context;
		if (!validation_phase.empty()) {
			LOG(INFO) << "Graph has validation phase (name: " << validation_phase << ")";
			validation_context = std::make_shared<ExecutionContext>();
			validation_context->current_epoch = 1;
			validation_context->debug_level = debug_level;
			validation_context->phase = validation_phase;
			validation_context->phase_behaviour = deepflow::PhaseParam_PhaseBehaviour_VALIDATION;
			for (auto node : _nodes) {
				if (node->includePhase(validation_phase) == false)
					continue;
				if (node->isGenerator())
					validation_phase_generators.push_back(node);
			}			
			validation_end_nodes = _get_end_nodes(validation_phase);
			validation_head_nodes = _get_head_nodes(validation_phase);
		}
		int iteration = 1;
		for (int epoch = 1; epoch <= max_epoch; ++epoch) {
			if (print_epoch) 
				std::cout << "Epoch " << epoch << " -->" << std::endl;
			auto epoch_start = std::chrono::high_resolution_clock::now();
			execution_context->current_epoch = epoch;
			_execute_one_pass(execution_context, &iteration, &_nodes, &execution_phase_generators, &execution_end_nodes, &execution_head_nodes, &execution_phase_variable_nodes, max_iter, print_iteration);
			if (!validation_phase.empty()) {
				validation_context->current_iteration = 1;
				validation_context->current_epoch = epoch;
				_execute_one_pass(validation_context, 0, &_nodes, &validation_phase_generators, &validation_end_nodes, &validation_head_nodes, 0, max_iter, false);
			}
			auto epoch_end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed_epoch = epoch_end - epoch_start;
			if (print_epoch)
				std::cout << "<-- Epoch " << epoch << " Elapsed time: " << elapsed_epoch.count() << " seconds" << std::endl;
			if (execution_context->quit == true)
				break;
			if (max_iter > 0 && iteration >= max_iter)
				break;
		}
	}
	else {
		_execute_one_pass(execution_context, 0, &_nodes, &execution_phase_generators, &execution_end_nodes, &execution_head_nodes, 0, max_iter, false);
	}
}

*/

void Session::mem_usage(size_t * free_byte, size_t * total_byte, float *used_byte_percentage)
{
	cudaError_t cuda_status = cudaMemGetInfo(free_byte, total_byte);
	if (cudaSuccess != cuda_status)
		LOG(FATAL) << "cudaMemGetInfo fails, " << cudaGetErrorString(cuda_status);
	if (used_byte_percentage)
		*used_byte_percentage = ((float)*total_byte - (float)*free_byte) / (*total_byte) * 100;
}

void generate_cpp_code(Node *node, std::string *code) {	
	(*code) += node->to_cpp() + "\n";	
}

std::string Session::to_cpp() const
{
	std::list<std::shared_ptr<Node>> ends = _get_all_end_nodes("");
	std::string code = "\nDeepFlow df;\n\n";
	
	for (auto phase : _phases)			
		code += "df.define_phase(\"" + phase.first + "\", deepflow::PhaseParam_PhaseBehaviour_" + PhaseParam_PhaseBehaviour_Name(phase.second) + ");\n";
	
	if (_phases.size() > 0)
		code += "\n";

	std::set<std::shared_ptr<Solver>> solvers_set;
	for (auto solver_map_item : _solvers) {
		bool exist = false;
		for (auto solver : solvers_set) {
			if (solver->name() == solver_map_item.second->name())
			{
				exist = true;
				break;
			}
		}
		if (!exist)
			solvers_set.insert(solver_map_item.second);
	}		
	for (auto solver : solvers_set)
		code += solver->to_cpp() + "\n";
	
	if (solvers_set.size() > 0)
		code += "\n";

	std::function<void(Node*)> foo = std::bind(generate_cpp_code, std::placeholders::_1, &code);
	
	int token = rand();
	//for (auto end : ends)
	//	end->_traverse_up(foo, Node::POST_ORDER, token);
	
	code += "\n";
	
	return code;
}

std::shared_ptr<PlaceHolder> Session::get_placeholder(std::string name)
{
	auto node = _find_node_by_name(name);
	LOG_IF(FATAL, node == nullptr) << "Node " << name << " does not exist.";
	auto placeholder = std::dynamic_pointer_cast<PlaceHolder>(node);
	LOG_IF(FATAL, placeholder == nullptr) << name << " is not a placeholder.";
	return placeholder;
}

std::shared_ptr<Node> Session::get_node(std::string name)
{
	auto node = _find_node_by_name(name);
	LOG_IF(FATAL, node == nullptr) << "Node " << name << " does not exist.";
	return node;
}

/*
std::shared_ptr<NetworkParam> DeepFlow::create_network_param(bool include_weights, bool include_inits) {
	auto param = std::make_shared<NetworkParam>();
	for (auto phase : _phases) {
		PhaseParam *phaseParam = param->add_phase();
		phaseParam->set_phase(phase.first);
		phaseParam->set_behaviour(phase.second);
	}
	for (auto var : _variables) {
		if (include_weights)
			var->transferDataToParam();
		else
			var->param().mutable_variable_param()->clear_weights();
		if (!include_inits)
			var->param().mutable_variable_param()->mutable_init_param()->clear_init_data();
	}
	for (auto node : _nodes) {
		NodeParam *nodeParam = param->add_node();
		nodeParam->CopyFrom(node->param());
	}
	SolverParam *solverParam = param->add_solver();
	solverParam->CopyFrom(_solver->param());
	return param;
}
*/