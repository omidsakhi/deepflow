#include "core/deep_flow.h"

#include "observers/reset.h"
#include "observers/forward.h"

#include <vector>

#include <fstream>

std::shared_ptr<MNISTReader> DeepFlow::mnist_reader(std::string folder_path, int batch_size, MNISTReaderType type, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	MnistReaderParam *param = nodeParam.mutable_mnist_reader_param();
	param->set_folder_path(folder_path);
	param->set_type((MnistReaderParam::ReaderType) type);
	param->set_batch_size(batch_size);
	auto node = std::make_shared<MNISTReader>(nodeParam);
	node->createIO();
	_nodes.push_back(node);
	return node;

}
std::shared_ptr<OutputTerminal> DeepFlow::add(std::shared_ptr<OutputTerminal> a, std::shared_ptr<OutputTerminal> b, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));	
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

std::shared_ptr<OutputTerminal> DeepFlow::bias_add(std::shared_ptr<OutputTerminal> a, std::shared_ptr<OutputTerminal> b, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	BiasAddParam *param = nodeParam.mutable_bias_add_param();
	auto node = std::make_shared<BiasAdd>(nodeParam);
	node->createIO();
	node->input(0)->connect(a);
	node->input(1)->connect(b);
	_nodes.push_back(node);
	return node->output(0);
}

std::shared_ptr<OutputTerminal> DeepFlow::subtract(std::shared_ptr<OutputTerminal> a, std::shared_ptr<OutputTerminal> b, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
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

std::shared_ptr<OutputTerminal> DeepFlow::softmax(std::shared_ptr<OutputTerminal> a, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	SoftmaxParam *param = nodeParam.mutable_softmax_param();
	param->set_alpha(1.0f);
	param->set_beta(0.0f);
	auto node = std::make_shared<Softmax>(nodeParam);
	node->createIO();
	node->input(0)->connect(a);
	_nodes.push_back(node);
	return node->output(0);
}

std::shared_ptr<OutputTerminal> DeepFlow::square(std::shared_ptr<OutputTerminal> a, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	SquareParam *param = nodeParam.mutable_square_param();	
	auto node = std::make_shared<Square>(nodeParam);
	node->createIO();
	node->input(0)->connect(a);
	_nodes.push_back(node);
	return node->output(0);
}

std::shared_ptr<OutputTerminal> DeepFlow::place_holder(std::array<int, 4> dims, Tensor::TensorType type, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
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

std::shared_ptr<OutputTerminal> DeepFlow::variable(std::shared_ptr<Initializer> initializer, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	VariableParam *varParam = nodeParam.mutable_variable_param();
	InitParam *initParam = varParam->mutable_init_param();
	initParam->CopyFrom(initializer->param());
	auto node = std::make_shared<Variable>(initializer, nodeParam);
	node->createIO();
	_nodes.push_back(node);
	_variables.push_back(node);
	return node->output(0);
}

std::shared_ptr<OutputTerminal> DeepFlow::variable(std::shared_ptr<Initializer> initializer, int interval, std::string prefix, int perImageHeight, int perImageWidth, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
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

std::shared_ptr<OutputTerminal> DeepFlow::matmul(std::shared_ptr<OutputTerminal> a, std::shared_ptr<OutputTerminal> b, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
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

std::shared_ptr<OutputTerminal> DeepFlow::relu(std::shared_ptr<OutputTerminal> a, float negative_slope, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	ReluParam *param = nodeParam.mutable_relu_param();
	param->set_negative_slope(negative_slope);
	auto node = std::make_shared<Relu>(nodeParam);
	node->createIO();
	node->input(0)->connect(a);
	_nodes.push_back(node);
	return node->output(0);
}

std::shared_ptr<Block> DeepFlow::block(std::initializer_list<std::shared_ptr<InputTerminal>> inputs, std::initializer_list<std::shared_ptr<OutputTerminal>> outputs, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	BlockParam *param = nodeParam.mutable_block_param();	
	auto node = std::make_shared<Block>(inputs, outputs, nodeParam);
	_nodes.push_back(node);
	return node;
}

std::shared_ptr<OutputTerminal> DeepFlow::softmax_loss(std::shared_ptr<OutputTerminal> a, std::shared_ptr<OutputTerminal> b, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
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

std::shared_ptr<SGDSolver> DeepFlow::sgd_solver(std::shared_ptr<OutputTerminal> loss, int max_iteration, float momentum, float learning_rate) {
	SolverParam param;
	param.set_max_iteration(max_iteration);
	SGDSolverParam *sp = param.mutable_sgd_solver();
	sp->set_learning_rate(learning_rate);
	sp->set_momentum(momentum);
	auto solver = std::make_shared<SGDSolver>(loss, param);
	_solvers.push_back(solver);
	return solver;
}

std::shared_ptr<GainSolver> DeepFlow::gain_solver(std::shared_ptr<OutputTerminal> loss, int max_iteration, float momentum, float learning_rate, float max_gain, float min_gain, float gain_plus, float gain_mult) {
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
	_solvers.push_back(solver);
	return solver;
}

std::shared_ptr<OutputTerminal> DeepFlow::dropout(std::shared_ptr<OutputTerminal> a, float dropout, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	DropoutParam *param = nodeParam.mutable_dropout_param();
	param->set_dropout(dropout);
	auto node = std::make_shared<Dropout>(nodeParam);
	node->createIO();
	node->input(0)->connect(a);
	_nodes.push_back(node);
	return node->output(0);
}

std::shared_ptr<OutputTerminal> DeepFlow::conv2d(std::shared_ptr<OutputTerminal> input, std::shared_ptr<OutputTerminal> filter, int pad_top_bottom, int pad_left_right, int vertical_filter_stride, int horizontal_filter_stride, int filter_height_dilation, int filter_width_dialation, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
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

std::shared_ptr<OutputTerminal> DeepFlow::conv2d(std::shared_ptr<OutputTerminal> input, std::shared_ptr<OutputTerminal> filter, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
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

std::shared_ptr<OutputTerminal> DeepFlow::pooling(std::shared_ptr<OutputTerminal> input, int windowHeight, int windowWidth, int verticalPadding, int horizontalPadding, int verticalStride, int horizontalStride, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
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

std::shared_ptr<OutputTerminal> DeepFlow::argmax(std::shared_ptr<OutputTerminal> input, int reduceDimension, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	ReduceParam *param = nodeParam.mutable_reduce_param();
	param->set_reduce_dim(reduceDimension);
	param->set_reduce_op(ReduceParam_ReduceOp_MAX);
	auto node = std::make_shared<Reduce>(nodeParam);
	node->createIO();
	node->input(0)->connect(input);
	_nodes.push_back(node);
	return node->output(1);
}

std::shared_ptr<OutputTerminal> DeepFlow::reduce_max(std::shared_ptr<OutputTerminal> input, int reduceDimension, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	ReduceParam *param = nodeParam.mutable_reduce_param();
	param->set_reduce_dim(reduceDimension);
	param->set_reduce_op(ReduceParam_ReduceOp_MAX);
	auto node = std::make_shared<Reduce>(nodeParam);
	node->createIO();
	node->input(0)->connect(input);
	_nodes.push_back(node);
	return node->output(0);
}

std::shared_ptr<OutputTerminal> DeepFlow::reduce_sum(std::shared_ptr<OutputTerminal> input, int reduceDimension, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	ReduceParam *param = nodeParam.mutable_reduce_param();
	param->set_reduce_dim(reduceDimension);
	param->set_reduce_op(ReduceParam_ReduceOp_ADD);
	auto node = std::make_shared<Reduce>(nodeParam);
	node->createIO();
	node->input(0)->connect(input);
	_nodes.push_back(node);
	return node->output(0);
}

std::shared_ptr<OutputTerminal> DeepFlow::reduce_mean(std::shared_ptr<OutputTerminal> input, int reduceDimension, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	ReduceParam *param = nodeParam.mutable_reduce_param();
	param->set_reduce_dim(reduceDimension);
	param->set_reduce_op(ReduceParam_ReduceOp_AVG);
	auto node = std::make_shared<Reduce>(nodeParam);
	node->createIO();
	node->input(0)->connect(input);
	_nodes.push_back(node);
	return node->output(0);
}

std::shared_ptr<OutputTerminal> DeepFlow::equal(std::shared_ptr<OutputTerminal> a, std::shared_ptr<OutputTerminal> b, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
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

void DeepFlow::eval(std::shared_ptr<OutputTerminal> terminal, bool restart) {		
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

std::tuple<float, float,int> DeepFlow::run(std::shared_ptr<OutputTerminal> loss, std::shared_ptr<OutputTerminal> accuracy, std::map<std::shared_ptr<OutputTerminal>, std::shared_ptr<OutputTerminal>> feed) {
	
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

void DeepFlow::saveAsBinary(std::string filePath) {
	NetworkParam netParam;
	for (auto var : _variables) {
		var->transferDataToParam();
	}
	for (auto node : _nodes) {		
		NodeParam *nodeParam = netParam.add_node();
		nodeParam->CopyFrom(node->param());
	}
	for (auto solver : _solvers) {
		SolverParam *solverParam = netParam.add_solver();
		solverParam->CopyFrom(solver->param());
	}
	std::fstream output(filePath, std::ios::out | std::ios::trunc | std::ios::binary);
	LOG_IF(FATAL, !netParam.SerializeToOstream(&output)) << "Failed to write network.";
}

#include <google/protobuf/text_format.h>

void DeepFlow::saveAsString(std::string filePath) {
	NetworkParam netParam;
	for (auto var : _variables) {
		var->transferDataToParam();
	}
	for (auto node : _nodes) {
		NodeParam *nodeParam = netParam.add_node();
		nodeParam->CopyFrom(node->param());
	}
	for (auto solver : _solvers) {
		SolverParam *solverParam = netParam.add_solver();
		solverParam->CopyFrom(solver->param());
	}
	std::string text;
	google::protobuf::TextFormat::PrintToString(netParam,&text);	
	std::ofstream out(filePath);
	out << text;
	out.close();
}