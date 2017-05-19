#include "core/caffe.h"

Caffe::Caffe(DeepFlow * df, bool verbose)
{
	this->df = df;
	_verbose = verbose;
}

void Caffe::load(std::string file_path, std::initializer_list<std::pair<std::string, std::array<int, 4>>> inputs)
{
	auto net = std::make_shared<caffe::NetParameter>();
	std::fstream input(file_path, std::ios::in | std::ios::binary);
	LOG_IF(FATAL, !net->ParseFromIstream(&input)) << "Failed to read caffe model " << file_path;
	input.close();
	if (net->layer_size() > 0) {
		LOG(INFO) << "Loading Caffe model from LayerParameter (NOT IMPLEMENTED)";
	}
	else if (net->layers_size() > 0) {
		_parse_net_deprecated(net, inputs);
	}
}

void Caffe::_parse_net_deprecated(std::shared_ptr<caffe::NetParameter> net, std::initializer_list<std::pair<std::string, std::array<int, 4>>> inputs)
{
	LOG_IF(INFO, _verbose) << "Loading Caffe model from V1LayerParameter (Deprecated) ...";	
	LOG_IF(INFO, _verbose) << "net.layers_size = " << net->layers_size();
	LOG_IF(INFO, _verbose) << "net.layer_size = " << net->layer_size();
	LOG_IF(INFO, _verbose) << "net.name = " << net->name();
	LOG_IF(INFO, _verbose) << "net.input_shape_size = " << net->input_shape_size();	
	LOG_IF(INFO, _verbose) << "net.input_dim_size = " << net->input_dim_size();		
	LOG_IF(INFO, _verbose) << "net.input_shape_size = " << net->input_shape_size();	
	LOG_IF(INFO, _verbose) << "net.force_backward = " << net->force_backward();	
	LOG_IF(INFO, _verbose) << "net.input_size = " << net->input_size();	
	for (auto net_input_name : net->input()) {
		LOG_IF(INFO, _verbose) << "    input = " << net_input_name;
		bool resolved = false;
		for (auto place_holde : inputs) {
			if (place_holde.first == net_input_name) {
				auto place_holder_node_output = df->place_holder(place_holde.second, Tensor::Float, net_input_name, {});
				auto place_holder_node_param = df->block()->find_node_param_by_output_name(place_holder_node_output);
				place_holder_node_param->set_output(0, net_input_name);
				resolved = true;
				break;
			}
		}
		LOG_IF(FATAL, resolved == false) << net_input_name << " input is not provided for the loading function.";
	}	
	
	for (auto layer : net->layers())
		_parse_layer_deprecated(layer);

	if (_inplace_nodes.size() > 0) {
		LOG_IF(INFO, _verbose) << "-> In-Place Nodes";
		for (auto node : _inplace_nodes)
			LOG_IF(INFO, _verbose) << "    " << node;
		_fix_inplace_nodes();
	}
}

void Caffe::_parse_layer_deprecated(const caffe::V1LayerParameter & layer)
{
	LOG_IF(INFO, _verbose) << "Type = " << caffe::V1LayerParameter_LayerType_Name(layer.type());
	LOG_IF(INFO, _verbose) << " .name = " << layer.name();	
	LOG_IF(FATAL, layer.name().empty()) << "layer.name().empty()";
	int bottom_size = layer.bottom_size();
	LOG_IF(INFO, _verbose) << " .bottom_size = " << bottom_size;
	for (auto b : layer.bottom())
		LOG_IF(INFO, _verbose) << "    .bottom = " << b;
	int top_size = layer.top_size();
	LOG_IF(INFO, _verbose) << " .top_size = " << top_size;
	for (auto t : layer.top())
		LOG_IF(INFO, _verbose) << "    .top = " << t;
	if (layer.include_size() > 0) {
		LOG_IF(INFO, _verbose) << " .include_size = " << layer.include_size();
	}
	if (layer.has_data_param()) {
		LOG_IF(INFO, _verbose) << " .data_param = true";
	}
	if (layer.blobs_size() > 0) {
		LOG_IF(INFO, _verbose) << " .blob_size = " << layer.blobs_size();
		for (auto blob : layer.blobs())
			_parse_blob_param(blob);
	}
	std::string df_node_output;
	if (layer.has_convolution_param()) {
		df_node_output = _parse_conv_param(layer.convolution_param(), layer);
	}	
	if (layer.has_relu_param()) {
		df_node_output = _parse_relu_param(layer.relu_param(), layer);
	}
	else if (layer.type() == caffe::V1LayerParameter::RELU) {
		caffe::ReLUParameter param;
		param.set_negative_slope(0.01f);
		df_node_output = _parse_relu_param(param, layer);
	}
	if (layer.has_pooling_param()) {
		df_node_output = _parse_pooling_param(layer.pooling_param(), layer);
	}
	if (layer.has_inner_product_param()) {
		df_node_output = _parse_inner_product_param(layer.inner_product_param(), layer);
	}
	if (layer.has_softmax_param()) {
		df_node_output = _parse_softmax_param(layer.softmax_param(), layer);
	}
	else if (layer.type() == caffe::V1LayerParameter::SOFTMAX) {
		df_node_output = _parse_softmax_param(caffe::SoftmaxParameter(), layer);
	}
	if (layer.has_dropout_param()) {
		df_node_output = _parse_dropout_param(layer.dropout_param(), layer);
	}
	else if (layer.type() == caffe::V1LayerParameter::DROPOUT) {
		caffe::DropoutParameter param;
		param.set_dropout_ratio(0.5);
		df_node_output = _parse_dropout_param(param, layer);
	}
	if (!df_node_output.empty() && top_size == bottom_size && top_size == 1 && layer.top(0) == layer.bottom(0)) {
		auto df_node = df->block()->find_node_param_by_output_name(df_node_output);
		_inplace_nodes.push_back(df_node->name());
	}
}

std::string Caffe::_parse_filler_param(std::initializer_list<int> dims, const caffe::FillerParameter & param, std::string name)
{
	// LOG_IF(INFO, _verbose) << " . = " << param. << " [default: ]";
	LOG_IF(INFO, _verbose) << "  -> FillerParameter " << name;
	LOG_IF(INFO, _verbose) << "     .type = " << param.type() << " [default: constant]";
	LOG_IF(INFO, _verbose) << "     .value = " << param.value() << " [default: 0]"; // constant
	LOG_IF(INFO, _verbose) << "     .min = " << param.min() << " [default: 0]"; // uniform
	LOG_IF(INFO, _verbose) << "     .max = " << param.max() << " [default: 1]"; // uniform
	LOG_IF(INFO, _verbose) << "     .mean = " << param.mean() << " [default: 0]"; // gaussian
	LOG_IF(INFO, _verbose) << "     .std = " << param.std() << " [default: 1]"; // gaussian
	LOG_IF(INFO, _verbose) << "     .sparse = " << param.sparse() << " [default: -1]";
	LOG_IF(INFO, _verbose) << "     .variance_norm = " << caffe::FillerParameter_VarianceNorm_Name(param.variance_norm()) << " [default: FAN_IN]"; // xavier and msra

	if (param.type() == "constant") {
		return df->fill(dims, param.value());
	}
	else if (param.type() == "uniform") {
		return df->random_uniform(dims, param.min(), param.max());
	}
	else if (param.type() == "gaussian" || param.type() == "xavier" || param.type() == "msra") {
		return df->random_normal(dims, param.mean(), param.std());
	}
	else {
		LOG(FATAL);
	}	
}

void Caffe::_parse_blob_param(const caffe::BlobProto & param)
{
	LOG_IF(INFO, _verbose) << "  -> BlobProto";
	LOG_IF(INFO, _verbose) << "     .data_size = " << param.data_size();
	LOG_IF(INFO, _verbose) << "     .diff_size = " << param.diff_size();	
	LOG_IF(INFO, _verbose) << "     .double_data_size = " << param.double_data_size();
	LOG_IF(INFO, _verbose) << "     .double_diff_size = " << param.double_diff_size();
}

void Caffe::_fix_inplace_nodes()
{
	while (!_inplace_nodes.empty()) {
		std::string inplace_node_name = _inplace_nodes.front();
		_inplace_nodes.pop_front();
		auto inplace_node_param = df->block()->find_node_param_by_name(inplace_node_name);
		auto inplace_node_input = inplace_node_param->input(0);
		auto inplace_node_output = inplace_node_param->output(0);
		df->block()->replace_node_params_input(inplace_node_input, inplace_node_output, { inplace_node_param });
	}
}

std::string Caffe::_parse_relu_param(const caffe::ReLUParameter & param, const caffe::V1LayerParameter &layer)
{
	LOG_IF(INFO, _verbose) << "  -> ReLUParameter";
	LOG_IF(INFO, _verbose) << "     .negative_slope = " << param.negative_slope() << " [default: 0]";
	return df->leaky_relu(layer.bottom(0) + "_output_0", param.negative_slope(), layer.name(), {});
}

std::string Caffe::_parse_dropout_param(const caffe::DropoutParameter & param, const caffe::V1LayerParameter &layer)
{
	LOG_IF(INFO, _verbose) << "  -> DropoutParameter";
	LOG_IF(INFO, _verbose) << "     .dropout_ratio = " << param.dropout_ratio() << " [default: 0.5]";
	return df->dropout(layer.bottom(0) + "_output_0", param.dropout_ratio(), true, layer.name(), {});
}

std::string Caffe::_parse_inner_product_param(const caffe::InnerProductParameter & param, const caffe::V1LayerParameter &layer)
{
	LOG_IF(INFO, _verbose) << "  -> InnerProductParameter";
	LOG_IF(INFO, _verbose) << "     .num_output = " << param.num_output();
	LOG_IF(INFO, _verbose) << "     .bias_term = " << param.bias_term() << " [default: true]";
	if (param.has_bias_filler()) {
		LOG_IF(INFO, _verbose) << "      .bias_filler = true";
		_parse_filler_param({}, param.bias_filler(), "bias_filler");
	}
	LOG_IF(INFO, _verbose) << "     .axis = " << param.axis() << " [default: 1]";
	LOG_IF(INFO, _verbose) << "     .transpose = " << param.transpose() << " [default: false]";

	int num_output = param.num_output();
	LOG_IF(FATAL, num_output < 1) << "num_output < 1";
	int num_inputs = layer.blobs(0).data_size() / num_output;
	//LOG_IF(FATAL, param.has_weight_filler() == false) << "param.has_weight_filler() == false - " << layer.name() ;
	std::string weight = df->variable( _parse_filler_param({ num_inputs, num_output, 1, 1 }, param.weight_filler(), "weight filler"), "", layer.name() + "_w", {});
	auto weights_node = df->block()->find_node_param_by_output_name(weight);
	auto weight_mutable_weights = weights_node->mutable_variable_param()->mutable_weights();
	for (auto d : layer.blobs(0).data())
		weight_mutable_weights->add_weight(d);
	if (param.bias_term() == false) {
		return df->matmul(layer.bottom(0) + "_output_0", weight, layer.name(), {});
	}
	else {
		//LOG_IF(FATAL, param.has_bias_filler() == false) << "param.has_bias_filler() == false - " << layer.name();
		std::string bias = df->variable(_parse_filler_param({ 1, num_output, 1, 1 }, param.bias_filler(), "bias filler"), "", layer.name() + "_b", {});
		auto bias_node = df->block()->find_node_param_by_output_name(bias);		
		auto bias_mutable_weights = bias_node->mutable_variable_param()->mutable_weights();
		for (auto d : layer.blobs(1).data())
			bias_mutable_weights->add_weight(d);		
		std::string m = df->matmul(layer.bottom(0) + "_output_0", weight, layer.name() + "_ip", {});
		return df->bias_add(m, bias, layer.name());
	}
}

std::string Caffe::_parse_conv_param(const caffe::ConvolutionParameter & param, const caffe::V1LayerParameter &layer)
{
	LOG_IF(INFO, _verbose) << "  -> ConvolutionParam";	

	LOG_IF(INFO, _verbose) << "     .axis = " << param.axis() << " [default: 1]";
	LOG_IF(FATAL, param.axis() != 1) << "Unsupported axis";

	LOG_IF(INFO, _verbose) << "     .kernel_size_size = " << param.kernel_size_size();
	LOG_IF(FATAL, param.kernel_size_size() > 1) << "Unsupported param.kernel_size_size() > 1";
	for (auto d : param.kernel_size())
		LOG_IF(INFO, _verbose) << "      .kernel_size = " << d;
	LOG_IF(INFO, _verbose) << "     .kernel_h = " << param.kernel_h();
	LOG_IF(INFO, _verbose) << "     .kernel_w = " << param.kernel_w();
	int kernel_h, kernel_w;
	if (param.kernel_size_size() == 1) {
		kernel_h = param.kernel_size(0);
		kernel_w = kernel_h;
	}
	else {
		LOG_IF(FATAL, param.kernel_h() == 0) << "No kernel_size or kernel_h is defined.";
		LOG_IF(FATAL, param.kernel_w() == 0) << "No kernel_size or kernel_w is defined.";
		kernel_h = param.kernel_h();
		kernel_w = param.kernel_w();
	}
	LOG_IF(INFO, _verbose) << "     .stride_size = " << param.stride_size() << " [default: 1]";
	for (auto d : param.stride())
		LOG_IF(INFO, _verbose) << "      .stride = " << d;
	LOG_IF(INFO, _verbose) << "     .stride_h = " << param.stride_h();
	LOG_IF(INFO, _verbose) << "     .stride_w = " << param.stride_w();
	int stride_h, stride_w;
	if (param.stride_size() == 1) {
		stride_h = param.stride(0);
		stride_w = stride_h;
	}
	else {
		stride_h = param.stride_h();
		stride_w = param.stride_w();
	}

	if (stride_h == 0)
		stride_h = 1;
	if (stride_w == 0)
		stride_w = 1;

	LOG_IF(INFO, _verbose) << "     .pad_size = " << param.pad_size() << " [default: 0]";
	for (auto d : param.pad())
		LOG_IF(INFO, _verbose) << "      .pad = " << d;
	LOG_IF(INFO, _verbose) << "     .pad_h = " << param.pad_h() << " [default: 0]";
	LOG_IF(INFO, _verbose) << "     .pad_w = " << param.pad_w() << " [default: 0]";
	int pad_h, pad_w;
	if (param.pad_size() == 1) {
		pad_h = param.pad(0);
		pad_w = pad_h;
	}
	else {
		pad_h = param.pad_h();
		pad_w = param.pad_w();
	}
	
	LOG_IF(INFO, _verbose) << "     .dilation_size = " << param.dilation_size() << " [default: 1]";
	for (auto d : param.dilation())
		LOG_IF(INFO, _verbose) << "      .dilation = " << d;
	int dilation_h, dilation_w;
	if (param.dilation_size() == 0) {
		dilation_h = dilation_w = 0;
	}
	else if (param.dilation_size() == 1) {
		dilation_h = dilation_w = param.dilation(0);
	}
	else if (param.dilation_size() == 2) {
		dilation_h = param.dilation(0);
		dilation_w = param.dilation(1);
	} else {
		LOG_IF(FATAL, param.dilation_size() > 2) << "Unsupported param.dilation_size = " << param.dilation_size();
	}

	if (dilation_h == 0)
		dilation_h = 1;
	if (dilation_w == 0)
		dilation_w = 1;
	
	if (param.has_weight_filler()) {
		LOG_IF(INFO, _verbose) << "     .weight_filler = true";		
	}
	//LOG_IF(FATAL, param.has_weight_filler() == false) << "param.has_weight_filler() == false - " << layer.name();
	LOG_IF(INFO, _verbose) << "     .num_output = " << param.num_output();
	int num_output = param.num_output();
	LOG_IF(FATAL, num_output < 1) << "num_output < 1";
	int filter_second_dimension = layer.blobs(0).data_size() / num_output / kernel_h / kernel_w;
	std::string filter = df->variable(_parse_filler_param({ num_output, filter_second_dimension, kernel_h, kernel_w }, param.weight_filler(), "weight filler"), "", layer.name() + "_w", {});
	auto filter_node = df->block()->find_node_param_by_output_name(filter);
	auto filter_mutable_weights = filter_node->mutable_variable_param()->mutable_weights();
	for (auto d : layer.blobs(0).data())
		filter_mutable_weights->add_weight(d);
	std::string bias;
	LOG_IF(INFO, _verbose) << "     .bias_term = " << param.bias_term();
	if (param.bias_term() == true) {
		if (param.has_bias_filler()) {
			LOG_IF(INFO, _verbose) << "     .bias_filler = true";
			_parse_filler_param({}, param.bias_filler(), "bias_filler");
		}
		//LOG_IF(FATAL, param.has_bias_filler() == false) << "param.has_bias_filler() == false - " << layer.name();
		bias = df->variable(_parse_filler_param({ 1, num_output, 1, 1 }, param.bias_filler(), "bias filler"), "", layer.name() + "_b", {});
		auto bias_node = df->block()->find_node_param_by_output_name(bias);
		auto bias_mutable_weights = bias_node->mutable_variable_param()->mutable_weights();
		for (auto d : layer.blobs(1).data())
			bias_mutable_weights->add_weight(d);
	}
	return df->conv2d(layer.bottom(0) + "_output_0", filter, bias, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, layer.name(), {});
}

std::string Caffe::_parse_softmax_param(const caffe::SoftmaxParameter & param, const caffe::V1LayerParameter & layer)
{
	LOG_IF(INFO, _verbose) << "  -> SoftmaxParameter";
	return df->softmax(layer.bottom(0) + "_output_0", layer.name(), {});
}

std::string Caffe::_parse_pooling_param(const caffe::PoolingParameter & param, const caffe::V1LayerParameter &layer)
{
	LOG_IF(INFO, _verbose) << "  -> PoolingParameter";
	LOG_IF(INFO, _verbose) << "     .pool_method = " << caffe::PoolingParameter_PoolMethod_Name(param.pool()) << " [default: MAX]";
	LOG_IF(INFO, _verbose) << "     .pad = " << param.pad() << " [default: 0]";
	LOG_IF(INFO, _verbose) << "     .pad_h = " << param.pad_h() << " [default: 0]";
	LOG_IF(INFO, _verbose) << "     .pad_w = " << param.pad_w() << " [default: 0]";
	LOG_IF(INFO, _verbose) << "     .kernel_size = " << param.kernel_size() << " [default: 0]";
	LOG_IF(INFO, _verbose) << "     .kernel_h = " << param.kernel_h();
	LOG_IF(INFO, _verbose) << "     .kernel_w = " << param.kernel_w();
	LOG_IF(INFO, _verbose) << "     .stride = " << param.stride() << " [default: 1]";
	LOG_IF(INFO, _verbose) << "     .stride_h = " << param.stride_h();
	LOG_IF(INFO, _verbose) << "     .stride_w = " << param.stride_w();
	LOG_IF(INFO, _verbose) << "     .global_pooling = " << param.global_pooling() << " [default: false]";
	
	int windowHeight, windowWidth;
	if (param.kernel_size() != 0) {
		windowHeight = param.kernel_size();
		windowWidth = param.kernel_size();
	}
	else {
		windowHeight = param.kernel_h();
		windowWidth = param.kernel_w();
	}
	int verticalPadding, horizontalPadding;
	if (param.pad() != 0) {
		verticalPadding = param.pad();
		horizontalPadding = param.pad();
	}
	else {
		verticalPadding = param.pad_h();
		horizontalPadding = param.pad_w();
	}
	int verticalStride, horizontalStride;
	if (param.stride() != 1) {
		verticalStride = param.stride();
		horizontalStride = param.stride();
	}
	else {
		verticalStride = param.stride_w();
		horizontalStride = param.stride_h();
	}

	return df->pooling(layer.bottom(0) + "_output_0", windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride, layer.name(), {});
}

