#include "core/deep_flow.h"

#include "observers/reset.h"
#include "observers/forward.h"

#include <vector>

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
	OpAddParam *param = nodeParam.mutable_op_add_param();
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
	OpBiasAddParam *param = nodeParam.mutable_op_bias_add_param();
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
	OpAddParam *param = nodeParam.mutable_op_add_param();
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
	OpSoftmaxParam *param = nodeParam.mutable_op_softmax_param();
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
	OpSquareParam *param = nodeParam.mutable_op_square_param();	
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
	TensorParam *tParam = nodeParam.mutable_tensor_param();
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
	VariableParam *param = nodeParam.mutable_variable_param();	
	auto node = std::make_shared<Variable>(initializer, nodeParam);
	node->createIO();
	_nodes.push_back(node);
	return node->output(0);
}

std::shared_ptr<OutputTerminal> DeepFlow::variable(std::shared_ptr<Initializer> initializer, int interval, std::string prefix, int perImageHeight, int perImageWidth, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	VariableParam *param = nodeParam.mutable_variable_param();
	SnapshotParam *snapshot = param->mutable_snapshot_param();
	snapshot->set_snapshot_interval(interval);
	snapshot->set_per_image_height(perImageHeight);
	snapshot->set_per_image_width(perImageWidth);
	snapshot->set_snapshot_prefix(prefix);
	auto node = std::make_shared<Variable>(initializer, nodeParam);
	node->createIO();
	_nodes.push_back(node);
	return node->output(0);
}

std::shared_ptr<OutputTerminal> DeepFlow::matmul(std::shared_ptr<OutputTerminal> a, std::shared_ptr<OutputTerminal> b, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	OpMatMulParam *param = nodeParam.mutable_op_matmul_param();
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
	OpReluParam *param = nodeParam.mutable_op_relu_param();
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
	return std::make_shared<SGDSolver>(loss, param);
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
	return std::make_shared<GainSolver>(loss, param);
}

std::shared_ptr<OutputTerminal> DeepFlow::dropout(std::shared_ptr<OutputTerminal> a, float dropout, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	OpDropoutParam *param = nodeParam.mutable_op_dropout_param();
	param->set_dropout(dropout);
	auto node = std::make_shared<Dropout>(nodeParam);
	node->createIO();
	node->input(0)->connect(a);
	_nodes.push_back(node);
	return node->output(0);
}

std::shared_ptr<OutputTerminal> DeepFlow::conv2d(std::shared_ptr<OutputTerminal> input, std::shared_ptr<OutputTerminal> filter, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	OpConv2dParam *param = nodeParam.mutable_op_conv_2d_param();
	param->set_pad_h(0);
	param->set_pad_w(0);
	param->set_u(1);
	param->set_v(1);
	param->set_upscale_x(1);
	param->set_upscale_y(1);
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
	OpPoolingParam *param = nodeParam.mutable_op_pooling_param();
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
	OpReduceParam *param = nodeParam.mutable_op_reduce_param();
	param->set_reduce_dim(reduceDimension);
	param->set_reduce_op(OpReduceParam_ReduceOp_MAX);
	auto node = std::make_shared<Reduce>(nodeParam);
	node->createIO();
	node->input(0)->connect(input);
	_nodes.push_back(node);
	return node->output(1);
}

std::shared_ptr<OutputTerminal> DeepFlow::reduce_max(std::shared_ptr<OutputTerminal> input, int reduceDimension, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	OpReduceParam *param = nodeParam.mutable_op_reduce_param();
	param->set_reduce_dim(reduceDimension);
	param->set_reduce_op(OpReduceParam_ReduceOp_MAX);
	auto node = std::make_shared<Reduce>(nodeParam);
	node->createIO();
	node->input(0)->connect(input);
	_nodes.push_back(node);
	return node->output(0);
}

std::shared_ptr<OutputTerminal> DeepFlow::reduce_sum(std::shared_ptr<OutputTerminal> input, int reduceDimension, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	OpReduceParam *param = nodeParam.mutable_op_reduce_param();
	param->set_reduce_dim(reduceDimension);
	param->set_reduce_op(OpReduceParam_ReduceOp_ADD);
	auto node = std::make_shared<Reduce>(nodeParam);
	node->createIO();
	node->input(0)->connect(input);
	_nodes.push_back(node);
	return node->output(0);
}

std::shared_ptr<OutputTerminal> DeepFlow::reduce_mean(std::shared_ptr<OutputTerminal> input, int reduceDimension, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	OpReduceParam *param = nodeParam.mutable_op_reduce_param();
	param->set_reduce_dim(reduceDimension);
	param->set_reduce_op(OpReduceParam_ReduceOp_AVG);
	auto node = std::make_shared<Reduce>(nodeParam);
	node->createIO();
	node->input(0)->connect(input);
	_nodes.push_back(node);
	return node->output(0);
}

std::shared_ptr<OutputTerminal> DeepFlow::equal(std::shared_ptr<OutputTerminal> a, std::shared_ptr<OutputTerminal> b, std::string name) {
	NodeParam nodeParam;
	nodeParam.set_name(getUniqueNodeName(name));
	OpEqualParam *equalParam = nodeParam.mutable_op_equal_param();
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

void DeepFlow::eval(std::shared_ptr<OutputTerminal> terminal) {
	ResetObserver resetObserver;
	ForwardObserver forwardObserver;	
	auto node = terminal->node();
	node->traverse(&resetObserver, TraverseOrder::PreOrder, true);
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
	int index = _nodes.size();
	std::string nodeName = prefix + "_" + std::to_string(index);
	while (findNodeByName(nodeName) != 0) {
		index++;
		nodeName = prefix + "_" + std::to_string(index);
	}
	return nodeName;
}