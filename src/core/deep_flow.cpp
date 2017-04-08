#include "core/deep_flow.h"

#include "observers/reset.h"
#include "observers/forward.h"

#include "core/variable.h"
#include "core/place_holder.h"
#include "core/phaseplexer.h"

#include "ops/add.h"
#include "ops/matmul.h"
#include "ops/relu.h"
#include "ops/softmax.h"
#include "ops/square.h"
#include "ops/bias_add.h"
#include "ops/softmax_loss.h"
#include "ops/dropout.h"
#include "ops/convolution_2d.h"
#include "ops/pooling.h"
#include "ops/reduce.h"
#include "ops/equal.h"

#include <unordered_map>
#include <map>

#include<memory>

#include <vector>

#include <fstream>

#include <chrono>

std::shared_ptr<MNISTReader> DeepFlow::mnist_reader(std::string folder_path, int batch_size, MNISTReaderType type, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	MnistReaderParam *param = nodeParam.mutable_mnist_reader_param();
	param->set_folder_path(folder_path);
	param->set_type((MnistReaderParam::ReaderType) type);
	param->set_batch_size(batch_size);
	auto node = std::make_shared<MNISTReader>(nodeParam);
	node->createIO();
	_nodes.push_back(node);
	return node;

}
NodeOutputPtr DeepFlow::add(NodeOutputPtr a, NodeOutputPtr b, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));	
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	AddParam *param = nodeParam.mutable_add_param();
	param->set_alpha(1.0f);
	param->set_beta(1.0f);
	auto node = std::make_shared<Add>(nodeParam);
	node->createIO();
	node->input(0)->connect(a);
	node->input(1)->connect(b);
	_nodes.push_back(node);
	return node->output(0);
}

NodeOutputPtr DeepFlow::bias_add(NodeOutputPtr a, NodeOutputPtr b, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	BiasAddParam *param = nodeParam.mutable_bias_add_param();
	auto node = std::make_shared<BiasAdd>(nodeParam);
	node->createIO();
	node->input(0)->connect(a);
	node->input(1)->connect(b);
	_nodes.push_back(node);
	return node->output(0);
}

NodeOutputPtr DeepFlow::subtract(NodeOutputPtr a, NodeOutputPtr b, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	AddParam *param = nodeParam.mutable_add_param();
	param->set_alpha(1.0f);
	param->set_beta(-1.0f);
	auto node = std::make_shared<Add>(nodeParam);
	node->createIO();
	node->input(0)->connect(a);
	node->input(1)->connect(b);
	_nodes.push_back(node);
	return node->output(0);
}

std::shared_ptr<IndexFill> DeepFlow::index_fill(std::initializer_list<int> dims, float offset, Tensor::TensorType type) {
	InitParam initParam;
	std::vector<int> values(dims);
	TensorParam *tParam = initParam.mutable_tensor_param();
	for (int i = 0; i < values.size(); ++i)
		tParam->add_dims(values[i]);
	tParam->set_type((TensorParam_TensorType)type);
	InitIndexFillParam *param = initParam.mutable_index_fill_param();
	param->set_offset(offset);
	return std::make_shared<IndexFill>(initParam);
}

std::shared_ptr<Step> DeepFlow::step(std::initializer_list<int> dims, float min, float max, Tensor::TensorType type) {
	InitParam initParam;
	std::vector<int> values(dims);
	TensorParam *tParam = initParam.mutable_tensor_param();
	for (int i = 0; i < values.size(); ++i)
		tParam->add_dims(values[i]);
	tParam->set_type((TensorParam_TensorType)type);
	InitStepParam *param = initParam.mutable_step_param();
	param->set_max(max);
	param->set_min(min);
	return std::make_shared<Step>(initParam);
}

std::shared_ptr<Fill> DeepFlow::fill(std::initializer_list<int> dims, float value, Tensor::TensorType type) {
	InitParam initParam;
	std::vector<int> values(dims);
	TensorParam *tParam = initParam.mutable_tensor_param();
	for (int i = 0; i < values.size(); ++i)
		tParam->add_dims(values[i]);
	tParam->set_type((TensorParam_TensorType)type);
	InitFillParam *param = initParam.mutable_fill_param();
	param->set_value(value);
	return std::make_shared<Fill>(initParam);
}

std::shared_ptr<Fill> DeepFlow::zeros(std::initializer_list<int> dims, Tensor::TensorType type) {
	InitParam initParam;
	std::vector<int> values(dims);
	TensorParam *tParam = initParam.mutable_tensor_param();
	for (int i = 0; i < values.size(); ++i)
		tParam->add_dims(values[i]);
	tParam->set_type((TensorParam_TensorType)type);
	InitFillParam *param = initParam.mutable_fill_param();
	param->set_value(0.0f);
	return std::make_shared<Fill>(initParam);
}

std::shared_ptr<Fill> DeepFlow::ones(std::initializer_list<int> dims, Tensor::TensorType type) {
	InitParam initParam;
	std::vector<int> values(dims);
	TensorParam *tParam = initParam.mutable_tensor_param();
	for (int i = 0; i < values.size(); ++i)
		tParam->add_dims(values[i]);
	tParam->set_type((TensorParam_TensorType)type);
	InitFillParam *param = initParam.mutable_fill_param();
	param->set_value(1.0f);
	return std::make_shared<Fill>(initParam);
}

std::shared_ptr<RandomUniform> DeepFlow::random_uniform(std::initializer_list<int> dims,  float min, float max, Tensor::TensorType type) {
	InitParam initParam;
	std::vector<int> values(dims);
	TensorParam *tParam = initParam.mutable_tensor_param();
	for (int i = 0; i < values.size(); ++i)
		tParam->add_dims(values[i]);
	tParam->set_type((TensorParam_TensorType)type);
	InitRandomUniformParam *param = initParam.mutable_random_uniform_param();
	param->set_max(max);
	param->set_min(min);
	return std::make_shared<RandomUniform>(initParam);
}

NodeOutputPtr DeepFlow::softmax(NodeOutputPtr a, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	SoftmaxParam *param = nodeParam.mutable_softmax_param();
	param->set_alpha(1.0f);
	param->set_beta(0.0f);
	auto node = std::make_shared<Softmax>(nodeParam);
	node->createIO();
	node->input(0)->connect(a);
	_nodes.push_back(node);
	return node->output(0);
}

NodeOutputPtr DeepFlow::square(NodeOutputPtr a, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	SquareParam *param = nodeParam.mutable_square_param();	
	auto node = std::make_shared<Square>(nodeParam);
	node->createIO();
	node->input(0)->connect(a);
	_nodes.push_back(node);
	return node->output(0);
}

NodeOutputPtr DeepFlow::place_holder(std::array<int, 4> dims, Tensor::TensorType type, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	PlaceHolderParam *param = nodeParam.mutable_place_holder_param();
	TensorParam *tParam = param->mutable_tensor_param();
	tParam->set_type((TensorParam_TensorType)type);
	for (int i=0; i<4; ++i)
		tParam->add_dims(dims[i]);
	auto node = std::make_shared<PlaceHolder>(nodeParam);
	node->createIO();
	_nodes.push_back(node);
	return node->output(0);
}

NodeOutputPtr DeepFlow::variable(std::shared_ptr<Initializer> initializer, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	VariableParam *varParam = nodeParam.mutable_variable_param();
	InitParam *initParam = varParam->mutable_init_param();
	initParam->CopyFrom(initializer->param());
	auto node = std::make_shared<Variable>(initializer, nodeParam);
	node->createIO();
	_nodes.push_back(node);
	_variables.push_back(node);
	return node->output(0);
}

NodeOutputPtr DeepFlow::variable(std::shared_ptr<Initializer> initializer, int interval, std::string prefix, int perImageHeight, int perImageWidth, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	VariableParam *varParam = nodeParam.mutable_variable_param();
	InitParam *initParam = varParam->mutable_init_param();
	initParam->CopyFrom(initializer->param());
	SnapshotParam *snapshot = varParam->mutable_snapshot_param();
	snapshot->set_snapshot_interval(interval);
	snapshot->set_per_image_height(perImageHeight);
	snapshot->set_per_image_width(perImageWidth);
	snapshot->set_snapshot_prefix(prefix);
	auto node = std::make_shared<Variable>(initializer, nodeParam);
	node->createIO();
	_nodes.push_back(node);
	_variables.push_back(node);
	return node->output(0);
}

NodeOutputPtr DeepFlow::matmul(NodeOutputPtr a, NodeOutputPtr b, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	MatMulParam *param = nodeParam.mutable_matmul_param();
	param->set_alpha(1.0f);
	param->set_beta(0.0f);
	auto node = std::make_shared<MatMul>(nodeParam);
	node->createIO();
	node->input(0)->connect(a);
	node->input(1)->connect(b);
	_nodes.push_back(node);
	return node->output(0);
}

NodeOutputPtr DeepFlow::relu(NodeOutputPtr a, float negative_slope, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	ReluParam *param = nodeParam.mutable_relu_param();
	param->set_negative_slope(negative_slope);
	auto node = std::make_shared<Relu>(nodeParam);
	node->createIO();
	node->input(0)->connect(a);
	_nodes.push_back(node);
	return node->output(0);
}

std::shared_ptr<Block> DeepFlow::block(std::initializer_list<NodeInputPtr> inputs, std::initializer_list<NodeOutputPtr> outputs, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	BlockParam *param = nodeParam.mutable_block_param();	
	auto node = std::make_shared<Block>(inputs, outputs, nodeParam);
	_nodes.push_back(node);
	return node;
}
std::shared_ptr<Print> DeepFlow::print(std::initializer_list<NodeOutputPtr> inputs, std::string message, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	PrintParam *printParam = nodeParam.mutable_print_param();
	printParam->set_message(message);
	printParam->set_num_inputs(inputs.size());
	auto node = std::make_shared<Print>(nodeParam);
	node->createIO();
	std::vector<NodeOutputPtr> inputsVec(inputs.size());
	std::copy(inputs.begin(), inputs.end(), inputsVec.begin());	
	for (int i=0; i < inputsVec.size();++i)
		node->input(i)->connect(inputsVec[i]);	
	_nodes.push_back(node);
	return node;
}

NodeOutputPtr DeepFlow::softmax_loss(NodeOutputPtr a, NodeOutputPtr b, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	LossParam *lossParam = nodeParam.mutable_loss_param();
	SoftmaxLossParam *param = lossParam->mutable_softmax_loss_param();
	param->set_alpha(1.0f);
	param->set_beta(0.0f);
	auto node = std::make_shared<SoftmaxLoss>(nodeParam);
	node->createIO();
	node->input(0)->connect(a);
	node->input(1)->connect(b);
	_nodes.push_back(node);
	return node->output(0);
}

std::shared_ptr<SGDSolver> DeepFlow::sgd_solver(NodeOutputPtr loss, int max_iteration, float momentum, float learning_rate) {
	SolverParam param;
	param.set_max_iteration(max_iteration);
	SGDSolverParam *sp = param.mutable_sgd_solver();
	sp->set_learning_rate(learning_rate);
	sp->set_momentum(momentum);
	auto solver = std::make_shared<SGDSolver>(loss, param);	
	return solver;
}

std::shared_ptr<GainSolver> DeepFlow::gain_solver(NodeOutputPtr loss, int max_iteration, float momentum, float learning_rate, float max_gain, float min_gain, float gain_plus, float gain_mult) {
	SolverParam param;
	param.set_max_iteration(max_iteration);
	GainSolverParam *gsp = param.mutable_gain_solver();
	gsp->set_momentum(momentum);
	gsp->set_learning_rate(learning_rate);
	gsp->set_max_gain(max_gain);
	gsp->set_min_gain(min_gain);
	gsp->set_gain_plus(gain_plus);
	gsp->set_gain_mult(gain_mult);    
	auto solver = std::make_shared<GainSolver>(loss, param);	
	return solver;
}

NodeOutputPtr DeepFlow::dropout(NodeOutputPtr a, float dropout, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	DropoutParam *param = nodeParam.mutable_dropout_param();
	param->set_dropout(dropout);
	auto node = std::make_shared<Dropout>(nodeParam);
	node->createIO();
	node->input(0)->connect(a);
	_nodes.push_back(node);
	return node->output(0);
}

NodeOutputPtr DeepFlow::conv2d(NodeOutputPtr input, NodeOutputPtr filter, int pad_top_bottom, int pad_left_right, int vertical_filter_stride, int horizontal_filter_stride, int filter_height_dilation, int filter_width_dialation, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	Conv2dParam *param = nodeParam.mutable_conv_2d_param();
	param->set_pad_h(pad_top_bottom);
	param->set_pad_w(pad_left_right);
	param->set_u(vertical_filter_stride);
	param->set_v(horizontal_filter_stride);
	param->set_dilation_h(filter_height_dilation);
	param->set_dilation_w(filter_width_dialation);
	auto node = std::make_shared<Convolution2D>(nodeParam);
	node->createIO();
	node->input(0)->connect(input);
	node->input(1)->connect(filter);
	_nodes.push_back(node);
	return node->output(0);
}

NodeOutputPtr DeepFlow::conv2d(NodeOutputPtr input, NodeOutputPtr filter, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	Conv2dParam *param = nodeParam.mutable_conv_2d_param();
	param->set_pad_h(0);
	param->set_pad_w(0);
	param->set_u(1);
	param->set_v(1);
	param->set_dilation_h(1);
	param->set_dilation_w(1);
	auto node = std::make_shared<Convolution2D>(nodeParam);
	node->createIO();
	node->input(0)->connect(input);
	node->input(1)->connect(filter);
	_nodes.push_back(node);
	return node->output(0);
}

NodeOutputPtr DeepFlow::pooling(NodeOutputPtr input, int windowHeight, int windowWidth, int verticalPadding, int horizontalPadding, int verticalStride, int horizontalStride, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	PoolingParam *param = nodeParam.mutable_pooling_param();
	param->set_h_pad(horizontalPadding);
	param->set_v_pad(verticalPadding);
	param->set_h_stride(horizontalStride);
	param->set_v_stride(verticalStride);
	param->set_window_h(windowHeight);
	param->set_window_w(windowWidth);
	auto node = std::make_shared<Pooling>(nodeParam);
	node->createIO();
	node->input(0)->connect(input);
	_nodes.push_back(node);
	return node->output(0);
}

NodeOutputPtr DeepFlow::argmax(NodeOutputPtr input, int reduceDimension, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	ReduceParam *param = nodeParam.mutable_reduce_param();
	param->set_reduce_dim(reduceDimension);
	param->set_reduce_op(ReduceParam_ReduceOp_MAX);
	auto node = std::make_shared<Reduce>(nodeParam);
	node->createIO();
	node->input(0)->connect(input);
	_nodes.push_back(node);
	return node->output(1);
}

NodeOutputPtr DeepFlow::reduce_max(NodeOutputPtr input, int reduceDimension, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	ReduceParam *param = nodeParam.mutable_reduce_param();
	param->set_reduce_dim(reduceDimension);
	param->set_reduce_op(ReduceParam_ReduceOp_MAX);
	auto node = std::make_shared<Reduce>(nodeParam);
	node->createIO();
	node->input(0)->connect(input);
	_nodes.push_back(node);
	return node->output(0);
}

NodeOutputPtr DeepFlow::phaseplexer(NodeOutputPtr input_1, std::string phase_1, NodeOutputPtr input_2, std::string phase_2, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	PhaseplexerParam *param = nodeParam.mutable_phaseplexer_param();
	param->add_phase(phase_1);
	param->add_phase(phase_2);
	auto node = std::make_shared<Phaseplexer>(nodeParam);
	node->createIO();
	node->input(0)->connect(input_1);
	node->input(1)->connect(input_2);
	_nodes.push_back(node);
	return node->output(0);
}

NodeOutputPtr DeepFlow::reduce_sum(NodeOutputPtr input, int reduceDimension, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	ReduceParam *param = nodeParam.mutable_reduce_param();
	param->set_reduce_dim(reduceDimension);
	param->set_reduce_op(ReduceParam_ReduceOp_ADD);
	auto node = std::make_shared<Reduce>(nodeParam);
	node->createIO();
	node->input(0)->connect(input);
	_nodes.push_back(node);
	return node->output(0);
}

NodeOutputPtr DeepFlow::reduce_mean(NodeOutputPtr input, int reduceDimension, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	ReduceParam *param = nodeParam.mutable_reduce_param();
	param->set_reduce_dim(reduceDimension);
	param->set_reduce_op(ReduceParam_ReduceOp_AVG);
	auto node = std::make_shared<Reduce>(nodeParam);
	node->createIO();
	node->input(0)->connect(input);
	_nodes.push_back(node);
	return node->output(0);
}

NodeOutputPtr DeepFlow::equal(NodeOutputPtr a, NodeOutputPtr b, std::string name, std::initializer_list<std::string> phases) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	for (auto phase : phases)
		nodeParam.add_phase(phase);
	EqualParam *equalParam = nodeParam.mutable_equal_param();
	auto node = std::make_shared<Equal>(nodeParam);
	node->createIO();
	node->input(0)->connect(a);
	node->input(1)->connect(b);
	_nodes.push_back(node);
	return node->output(0);
}

void DeepFlow::global_node_initializer() {
	std::list<std::shared_ptr<Node>> queue = _nodes;
	while (!queue.empty()) {
		auto node = queue.front();		
		queue.pop_front();
		if (node->isInitialized())
			continue;
		bool resolved = true;
		for(auto input : node->inputs()) {
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

void DeepFlow::eval(NodeOutputPtr terminal, bool restart) {		
	auto node = terminal->node();
	if (restart) {
		ResetObserver resetObserver;
		node->traverse(&resetObserver, TraverseOrder::PreOrder, true);
	}
	ForwardObserver forwardObserver;
	node->traverse(&forwardObserver, TraverseOrder::PostOrder, false);
}

std::shared_ptr<Node> DeepFlow::findNodeByName(const std::string &name) const {
	for (auto node : _nodes) {
		if (node->name() == name)
			return node;
	}
	return 0;
}

std::string DeepFlow::getUniqueNodeName(const std::string &prefix) const {
	if (findNodeByName(prefix) == 0)
		return prefix;
	int index = 1;
	std::string nodeName = prefix + "_" + std::to_string(index);
	while (findNodeByName(nodeName) != 0) {
		index++;
		nodeName = prefix + "_" + std::to_string(index);
	}
	return nodeName;
}

void DeepFlow::run(std::string phase, bool print_iteration, bool print_epoch) {
	LOG_IF(FATAL, _phases.find(phase) == _phases.end()) << "Specified phase " << phase << " is not defined.";
	PhaseParam_PhaseBehaviour behaviour = _phases.find(phase)->second;
	std::list<std::shared_ptr<Reader>> readers;
	std::list<std::shared_ptr<Node>> end_nodes;
	for (auto node : _nodes) {
		node->setExecutionPhase(phase);
		if (node->includePhase(phase)) {
			auto reader = std::dynamic_pointer_cast<Reader>(node);
			if (reader) {
				readers.push_back(reader);
			}		
			int num_connected_outputs = 0;
			for (auto output : node->outputs()) {
				if (output->connectedNode())
					num_connected_outputs++;
			}
			if (num_connected_outputs == 0)
				end_nodes.push_back(node);
		}
	}		
	LOG_IF(FATAL, readers.size() == 0) << "No reader is defined for phase " << phase;
	if (behaviour == PhaseParam_PhaseBehaviour_TRAIN) {
		LOG_IF(FATAL, _solver == 0) << "No solver is defined for the graph.";
		NodePtr loss_nodes;
		int count = 0;
		for (auto node : _nodes) {
			auto loss = std::dynamic_pointer_cast<Loss>(node);
			if (loss && loss->includePhase(phase)) {
				count = 0;
				loss_nodes = node;
			}				
		}
		LOG_IF(FATAL, loss_nodes = 0) << "No loss node is defined for phase " << phase;
		LOG_IF(WARNING, count > 1) << "More than one loss node is defined for phase " << phase;
		
		int iteration = 0;		
		ForwardObserver forwardObserver;
		ResetObserver resetObserver;
		for (int iteration = 1; iteration <= _solver->maxIteration(); ++iteration) {
			if (print_iteration)
				std::cout << "Iteration " << iteration << " -->" << std::endl;
			auto iteration_start = std::chrono::high_resolution_clock::now();
			int epoch = 1;
			bool any_last_batch = false;
			do {				
				if (print_epoch)
					std::cout << "  Epoch " << epoch << std::endl;
				for (auto node : end_nodes)
					node->traverse(&resetObserver, TraverseOrder::PreOrder, true);
				_solver->train_step();
				for (auto node : end_nodes)
					node->traverse(&forwardObserver, TraverseOrder::PostOrder, false);				
				for (auto reader : readers) {
					if (reader->isLastBatch())
						any_last_batch = true;
					reader->nextBatch();
				}				
				epoch++;
			} while (any_last_batch == false);
			auto iteration_end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed_iteration = iteration_end - iteration_start;
			if (print_iteration)
				std::cout << "<-- Iteration " << iteration << " Elapsed time: " << elapsed_iteration.count() << " seconds" << std::endl;		
		}
	}
}

std::tuple<float, float,int> DeepFlow::run(NodeOutputPtr loss, NodeOutputPtr accuracy, std::map<NodeOutputPtr, NodeOutputPtr> feed) {
	
	std::set<std::shared_ptr<Reader>> readers;	

	for (auto item : feed) {
		auto readerNode = item.second->node();
		auto reader = std::dynamic_pointer_cast<Reader>(readerNode);
		LOG_IF(FATAL, reader == 0) << readerNode->name() << " is not a reader.";
		readers.insert(reader);		
		auto placeHolderNode = item.first->node();		
		auto placeHolder = std::dynamic_pointer_cast<PlaceHolder>(placeHolderNode);
		LOG_IF(FATAL, placeHolder == 0) << placeHolderNode->name() << " is not a placeholder.";		
	}
		
	ResetObserver resetObserver;
	loss->node()->traverse(&resetObserver, TraverseOrder::PreOrder, true);
	accuracy->node()->traverse(&resetObserver, TraverseOrder::PreOrder, true);

	float totalAccuracy = 0;
	float totalLoss = 0;
	int totalBatch = 0;
	bool any_last_batch = false;
	do {
		for (auto item : feed)
			item.first->feed(item.second);
		if (loss) {
			eval(loss);
			totalLoss += loss->value()->toFloat();
		}		
		if (accuracy) {
			eval(accuracy, loss? false: true);
			totalAccuracy += accuracy->value()->toFloat();
		}				
		totalBatch++;
		for (auto reader : readers) {
			if (reader->isLastBatch())
				any_last_batch = true;
			reader->nextBatch();
		}
	} while (any_last_batch == false);
	totalAccuracy /= totalBatch;
	totalLoss /= totalBatch;	
	return std::make_tuple(totalLoss, totalAccuracy, totalBatch);
}

std::shared_ptr<NetworkParam> DeepFlow::createNetworkParam(bool include_weights, bool include_inits) {
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

void DeepFlow::save_as_binary(std::string filePath, bool include_inits) {
	auto param = createNetworkParam(true, include_inits);
	std::fstream output(filePath, std::ios::out | std::ios::trunc | std::ios::binary);
	LOG_IF(FATAL, !param->SerializeToOstream(&output)) << "Failed to write network.";
}

#include <google/protobuf/text_format.h>

void DeepFlow::save_as_text(std::string filePath, bool include_weights, bool include_inits) {
	auto param = createNetworkParam(include_weights,include_inits);
	std::string text;
	google::protobuf::TextFormat::PrintToString(*param,&text);	
	std::ofstream out(filePath);
	out << text;
	out.close();
}

void DeepFlow::define_phase(std::string phase, PhaseParam_PhaseBehaviour behaviour) {
	_phases.insert(std::pair<std::string, PhaseParam_PhaseBehaviour>(phase,behaviour));
}

void DeepFlow::set_solver(std::shared_ptr<Solver> solver) {
	_solver = solver;
}