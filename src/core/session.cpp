#include "core/session.h"

#include "nodes/variable.h"
#include "nodes/place_holder.h"
#include "nodes/phaseplexer.h"

#include "initializers/fill.h"
#include "initializers/index_fill.h"
#include "initializers/random_uniform.h"
#include "initializers/random_normal.h"
#include "initializers/step.h"

#include "core/solver.h"

#include "solvers/gain_solver.h"
#include "solvers/sgd_solver.h"
#include "solvers/adam_solver.h"
#include "solvers/adadelta_solver.h"

#include "nodes/add.h"
#include "nodes/matmul.h"
#include "nodes/leaky_relu.h"
#include "nodes/softmax.h"
#include "nodes/square.h"
#include "nodes/bias_add.h"
#include "nodes/softmax_loss.h"
#include "nodes/dropout.h"
#include "nodes/convolution_2d.h"
#include "nodes/pooling.h"
#include "nodes/reduce.h"
#include "nodes/equal.h"
#include "nodes/cast_float.h"
#include "nodes/display.h"
#include "nodes/transposed_conv_2d.h"
#include "nodes/euclidean_loss.h"
#include "nodes/activation.h"
#include "nodes/psnr.h"
#include "nodes/random_selector.h"

#include "generators/data_generator.h"
#include "generators/image_reader.h"

#include <unordered_map>
#include <map>

#include<memory>

#include <vector>

#include <fstream>

#include <chrono>

#include <ctime>

std::shared_ptr<Initializer> _create_initializer(const InitParam &init_param) {
	
	if (init_param.has_fill_param()) {
		return std::make_shared<Fill>(init_param);
	}
	else if (init_param.has_index_fill_param()) {
		return std::make_shared<IndexFill>(init_param);
	}
	else if (init_param.has_random_uniform_param()) {
		return std::make_shared<RandomUniform>(init_param);
	}
	else if (init_param.has_step_param()) {
		return std::make_shared<Step>(init_param);
	}
	else if (init_param.has_random_normal_param()) {
		return std::make_shared<RandomNormal>(init_param);
	}
	else {
		LOG(FATAL) << "Unsupported Initializer";
	}

	return NULL;
}

std::shared_ptr<Node> Session::_create_node(const NodeParam &node_param) {	

	if (node_param.has_accumulator_param())
		return std::make_shared<Accumulator>(node_param);	
	else if (node_param.has_generator_param()) {
		const GeneratorParam &generator_param = node_param.generator_param();
		if (generator_param.has_mnist_param()) {
			return std::make_shared<MNISTReader>(node_param);
		}
		else if (generator_param.has_data_generator_param()) {
			const InitParam &init_param = node_param.variable_param().init_param();
			std::shared_ptr<Initializer> initializer = _create_initializer(init_param);
			return std::make_shared<DataGenerator>(initializer,node_param);
		}
		else if (generator_param.has_image_reader_param()) {
			return std::make_shared<ImageReader>(node_param);
		}
	}
	else if (node_param.has_add_param())
		return std::make_shared<Add>(node_param);
	else if (node_param.has_bias_add_param())
		return std::make_shared<BiasAdd>(node_param);
	else if (node_param.has_cast_float_param())
		return std::make_shared<CastFloat>(node_param);
	else if (node_param.has_conv_2d_param())
		return std::make_shared<Convolution2D>(node_param);
	else if (node_param.has_dropout_param())
		return std::make_shared<Dropout>(node_param);
	else if (node_param.has_equal_param())
		return std::make_shared<Equal>(node_param);
	else if (node_param.has_loss_param()) {
		if (node_param.loss_param().has_softmax_loss_param())
			return std::make_shared<SoftmaxLoss>(node_param);
		else if (node_param.loss_param().has_euclidean_loss_param())
			return std::make_shared<EuclideanLoss>(node_param);
	}
	else if (node_param.has_activation_param()) {
		return std::make_shared<Activation>(node_param);
	}
	else if (node_param.has_matmul_param()) {
		return std::make_shared<MatMul>(node_param);
	}
	else if (node_param.has_phaseplexer_param()) {
		return std::make_shared<Phaseplexer>(node_param);
	}
	else if (node_param.has_place_holder_param()) {
		return std::make_shared<PlaceHolder>(node_param);
	}
	else if (node_param.has_pooling_param()) {
		return std::make_shared<Pooling>(node_param);
	}
	else if (node_param.has_print_param()) {
		return std::make_shared<Print>(node_param);
	}
	else if (node_param.has_display_param()) {
		return std::make_shared<Display>(node_param);
	}
	else if (node_param.has_reduce_param()) {
		return std::make_shared<Reduce>(node_param);
	}
	else if (node_param.has_leaky_relu_param()) {
		return std::make_shared<LeakyRelu>(node_param);
	}
	else if (node_param.has_softmax_param()) {
		return std::make_shared<Softmax>(node_param);
	}
	else if (node_param.has_square_param()) {
		return std::make_shared<Square>(node_param);
	}
	else if (node_param.has_transposed_conv_2d_param()) {
		return std::make_shared<TransposedConvolution2D>(node_param);
	}
	else if (node_param.has_psnr_param()) {
		return std::make_shared<Psnr>(node_param);
	}
	else if (node_param.has_random_selector_param()) {
		return std::make_shared<RandomSelector>(node_param);
	}
	else if (node_param.has_variable_param()) {
		const InitParam &init_param = node_param.variable_param().init_param();
		std::shared_ptr<Initializer> initializer = _create_initializer(init_param);
		return std::make_shared<Variable>(initializer, node_param);
	}
	else {
		LOG(FATAL) << "Unsupported Node";
	}

	return 0;
}

std::shared_ptr<Solver> Session::_create_solver(const SolverParam &solver_param) {
	if (solver_param.has_gain_solver()) {
		return std::make_shared<GainSolver>(solver_param);
	}
	else if (solver_param.has_sgd_solver()) {
		return std::make_shared<SGDSolver>(solver_param);
	}
	else if (solver_param.has_adam_solver()) {
		return std::make_shared<AdamSolver>(solver_param);
	}
	else if (solver_param.has_adadelta_solver()) {
		return std::make_shared<AdaDeltaSolver>(solver_param);
	}
	else {
		LOG(FATAL) << "Unsupported Solver";
	}

	return 0;
}

void Session::setGraph(std::shared_ptr<GraphParam> graph) {
	_graph = graph;
}

void Session::initialize() {
	
	for (auto phase_param : _graph->phase())
		_phases.insert(std::pair<std::string, PhaseParam_PhaseBehaviour>(phase_param.phase(), phase_param.behaviour()));

	for (auto node_param : _graph->node()) {
		auto node = _create_node(node_param);
		node->createIO();
		LOG_IF(FATAL, node->inputs().size() != node->param().input_size()) << "Node " << node->name() << "'s input size " << node->inputs().size() << " does not match the one specified in proto (" << node->param().input_size() << ")";
		_nodes.push_back(node);
	}

	for (auto node : _nodes) {
		for (int i = 0; i < node->param().input_size(); ++i) {
			const std::string terminal_name = node->param().input(i);
			auto terminal = _find_node_output_by_name(terminal_name);
			LOG_IF(FATAL, terminal == 0) << "Failed to find " << terminal_name << " for node " << node->name() ;
			node->input(i)->connect(terminal);
		}
	}
	
	_variables = _get_nodes<Variable>("");

	for (auto var : _variables) {
		std::shared_ptr<Solver> solver;
		std::string var_solver_name = var->param().variable_param().solver_name();
		for (auto solver_param : _graph->solver()) {
			if (var_solver_name == solver_param.name()) {
				solver = _create_solver(solver_param);
				break;
			}
		}
		if (solver) {
			_solvers.insert(std::pair<std::shared_ptr<Variable>, std::shared_ptr<Solver>>(var, solver));
			LOG(INFO) << "Variable " << var->name() << " <-> Solver " << solver->name();
		}
		else {
			LOG(INFO) << "Variable " << var->name() << " <-> Constant";
		}
	}

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
			node->initForward();
			node->initBackward();
			node->setInitialized(true);
		}
		else {
			queue.push_back(node);
		}
	}
}

std::shared_ptr<Node> Session::_find_node_by_name(const std::string &name) const {
	for (auto node : _nodes) {
		if (node->name() == name)
			return node;
	}
	return 0;
}

std::list<std::shared_ptr<Node>> Session::_get_end_nodes(std::string execution_phase)
{
	std::list<std::shared_ptr<Node>> list;
	for (auto node : _nodes) {
		if (node->includePhase(execution_phase)) {
			int num_connected_outputs = 0;
			for (auto output : node->outputs())
				num_connected_outputs += output->connectedNodes().size();
			if (num_connected_outputs == 0)
				list.push_back(node);
		}
	}
	return list;
}

std::shared_ptr<NodeOutput> Session::_find_node_output_by_name(const std::string &name) const {
	for (auto node : _nodes) {
		for (auto terminal : node->outputs()) {
			if (terminal->name() == name)
				return terminal;
		}
	}
	return 0;
}

void Session::_execute_one_pass(std::shared_ptr<ExecutionContext> context, int *iteration, std::list<std::shared_ptr<Node>> *nodes, std::list<std::shared_ptr<Generator>> *generators, std::list<std::shared_ptr<Node>> *end_nodes, std::list<std::shared_ptr<Loss>> *loss_nodes, std::list<std::shared_ptr<Variable>> *variable_nodes, int max_iter, bool train, bool print_iteration) {
	int iteration_per_epoch = 1;
	bool any_last_batch = false;
	for (auto node : *nodes)
		node->setExecutionContext(context);
	for (auto node : *end_nodes) {
		node->_unvisit();
		node->_shouldForward();
		if (train) {
			node->_unvisit();
			node->_shouldBackward();
		}
	}
	do {
		for (auto gen : *generators) {
			if (gen->isLastBatch())
				any_last_batch = true;
		}		
		if (iteration)
			context->current_iteration = *iteration;
		context->current_iteration_per_epoch = iteration_per_epoch;
		context->last_batch = any_last_batch;
		if (iteration && print_iteration)
			std::cout << "  Iteration " << *iteration << std::endl;
		for (auto node : *end_nodes) {
			node->_unvisit();
			node->_forward();
		}
		if (train) {
			for (auto node : *end_nodes) {
				node->_unvisit();
				node->_backward();
			}
			for (auto var : *variable_nodes) {
				auto map_var_to_solver = _solvers.find(var);
				if (map_var_to_solver != _solvers.end()) {					
					map_var_to_solver->second->apply(var);
				}
			}
		}
		for (auto gen : *generators)
			gen->nextBatch();
		if (iteration)
			(*iteration)++;
		iteration_per_epoch++;
		if (max_iter > 0 && iteration && (*iteration) >= max_iter)
			break;
	} while (any_last_batch == false && context->quit != true);
}

void Session::run(std::string phase, int max_epoch, int max_iter, bool print_iteration, bool print_epoch, int debug_level) {
	LOG(INFO) << "Executing graph for phase " << phase;
	LOG_IF(FATAL, _phases.find(phase) == _phases.end()) << "Specified phase " << phase << " is not defined.";
	PhaseParam_PhaseBehaviour behaviour = _phases.find(phase)->second;
	std::list<std::shared_ptr<Generator>> execution_phase_generators = _get_nodes<Generator>(phase);
	LOG_IF(FATAL, execution_phase_generators.size() == 0) << "No generator is defined for phase " << phase;
	std::list<std::shared_ptr<Node>> execution_end_nodes = _get_end_nodes(phase);
	std::string end_node_names;
	for (auto node : execution_end_nodes) {
		end_node_names += node->name() + " ";
	}
	LOG(INFO) << "End nodes: " << end_node_names;
	auto execution_context = std::make_shared<ExecutionContext>();
	execution_context->phase = phase;
	execution_context->debug_level = debug_level;
	if (behaviour == PhaseParam_PhaseBehaviour_TRAIN) {		
		std::list<std::shared_ptr<Variable>> execution_phase_variable_nodes = _get_nodes<Variable>(phase);
		std::list<std::shared_ptr<Loss>> execution_phase_loss_nodes = _get_nodes<Loss>(phase);
		LOG_IF(FATAL, execution_phase_loss_nodes.size() == 0) << "No loss node is defined for phase " << phase;
		LOG_IF(WARNING, execution_phase_loss_nodes.size() > 1) << "More than one loss node is defined for phase " << phase;
		std::string validation_phase;
		for (auto phase : _phases) {
			if (phase.second == PhaseParam_PhaseBehaviour_VALIDATION) {
				validation_phase = phase.first;
				break;
			}
		}
		std::list<std::shared_ptr<Generator>> validation_readers;
		std::list<std::shared_ptr<Node>> validation_end_nodes;
		std::shared_ptr<ExecutionContext> validation_context;
		if (!validation_phase.empty()) {
			LOG(INFO) << "Graph has validation phase (name: " << validation_phase << ")";
			validation_context = std::make_shared<ExecutionContext>();
			validation_context->current_epoch = 1;
			validation_context->debug_level = debug_level;
			validation_readers = _get_nodes<Generator>(validation_phase);
			validation_end_nodes = _get_end_nodes(validation_phase);
		}
		int iteration = 1;
		for (int epoch = 1; epoch <= max_epoch; ++epoch) {
			if (print_epoch)
				std::cout << "Epoch " << epoch << " -->" << std::endl;
			auto epoch_start = std::chrono::high_resolution_clock::now();
			execution_context->current_epoch = epoch;
			_execute_one_pass(execution_context, &iteration, &_nodes, &execution_phase_generators, &execution_end_nodes, &execution_phase_loss_nodes, &execution_phase_variable_nodes, max_iter, true, print_iteration);
			if (!validation_phase.empty()) {
				validation_context->phase = validation_phase;
				validation_context->current_iteration = 1;
				validation_context->current_epoch = epoch;
				_execute_one_pass(validation_context, 0, &_nodes, &validation_readers, &validation_end_nodes, 0, 0, max_iter, false, false);
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
		_execute_one_pass(execution_context, 0, &_nodes, &execution_phase_generators, &execution_end_nodes,0,0, max_iter, false, false);
	}
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