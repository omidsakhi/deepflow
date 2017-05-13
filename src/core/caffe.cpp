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
		for (auto place_holde : inputs) {
			if (place_holde.first == net_input_name) {
				auto place_holder_node_output = df->place_holder(place_holde.second, Tensor::Float, net_input_name, {});
				auto place_holder_node_param = df->block()->find_node_by_output__name(place_holder_node_output);
				place_holder_node_param->set_output(0, net_input_name);
			}
		}
	}	
	for (auto layer : net->layers())
		_parse_layer_deprecated(layer);	
}

void Caffe::_parse_layer_deprecated(const caffe::V1LayerParameter & layer)
{
	LOG_IF(INFO, _verbose) << "Type = " << caffe::V1LayerParameter_LayerType_Name(layer.type());
	LOG_IF(INFO, _verbose) << " .name = " << layer.name();	
	LOG_IF(FATAL, layer.name().empty()) << "layer.name().empty()";
	LOG_IF(INFO, _verbose) << " .bottom_size = " << layer.bottom_size();
	for (auto b : layer.bottom())
		LOG_IF(INFO, _verbose) << "    .bottom = " << b;
	LOG_IF(INFO, _verbose) << " .top_size = " << layer.top_size();
	for (auto t : layer.top())
		LOG_IF(INFO, _verbose) << "    .top = " << t;	
	if (layer.has_convolution_param()) {
		_parse_conv_param(layer.convolution_param(), layer);
	}	
	if (layer.has_relu_param()) {
		_parse_relu_param(layer.relu_param(), layer);
	}	
	if (layer.has_pooling_param()) {
		_parse_pooling_param(layer.pooling_param(), layer);
	}
	if (layer.has_data_param()) {
		LOG_IF(INFO, _verbose) << " .data_param = true";
	}
	if (layer.has_inner_product_param()) {
		_parse_inner_product_param(layer.inner_product_param(), layer);
	}
	if (layer.blobs_size() > 0) {
		LOG_IF(INFO, _verbose) << " .blob_size = " << layer.blobs_size();
		for (auto blob : layer.blobs())
			_parse_blob_param(blob);
	}
	if (layer.has_dropout_param()) {
		_parse_dropout_param(layer.dropout_param(), layer);
	}
	if (layer.include_size() > 0) {
		LOG_IF(INFO, _verbose) << " .include_size = " << layer.include_size();
	}	
}

std::shared_ptr<deepflow::InitParam> Caffe::_parse_filler_param(std::initializer_list<int> dims, const caffe::FillerParameter & _block_param, std::string name)
{
	// LOG_IF(INFO, _verbose) << " . = " << param. << " [default: ]";
	LOG_IF(INFO, _verbose) << "  -> FillerParameter " << name;
	LOG_IF(INFO, _verbose) << "     .type = " << _block_param.type() << " [default: constant]";
	LOG_IF(INFO, _verbose) << "     .value = " << _block_param.value() << " [default: 0]"; // constant
	LOG_IF(INFO, _verbose) << "     .min = " << _block_param.min() << " [default: 0]"; // uniform
	LOG_IF(INFO, _verbose) << "     .max = " << _block_param.max() << " [default: 1]"; // uniform
	LOG_IF(INFO, _verbose) << "     .mean = " << _block_param.mean() << " [default: 0]"; // gaussian
	LOG_IF(INFO, _verbose) << "     .std = " << _block_param.std() << " [default: 1]"; // gaussian
	LOG_IF(INFO, _verbose) << "     .sparse = " << _block_param.sparse() << " [default: -1]";
	LOG_IF(INFO, _verbose) << "     .variance_norm = " << caffe::FillerParameter_VarianceNorm_Name(_block_param.variance_norm()) << " [default: FAN_IN]"; // xavier and msra

	if (_block_param.type() == "constant") {
		return df->fill(dims, _block_param.value());
	}
	else if (_block_param.type() == "uniform") {
		return df->random_uniform(dims, _block_param.min(), _block_param.max());
	}
	else if (_block_param.type() == "gaussian" || _block_param.type() == "xavier" || _block_param.type() == "msra") {
		return df->random_normal(dims, _block_param.mean(), _block_param.std());
	}
	else {
		LOG(FATAL);
	}	
}

void Caffe::_parse_blob_param(const caffe::BlobProto & _block_param)
{
	LOG_IF(INFO, _verbose) << "  -> BlobProto";
	LOG_IF(INFO, _verbose) << "     .data_size = " << _block_param.data_size();
	LOG_IF(INFO, _verbose) << "     .diff_size = " << _block_param.diff_size();	
	LOG_IF(INFO, _verbose) << "     .double_data_size = " << _block_param.double_data_size();
	LOG_IF(INFO, _verbose) << "     .double_diff_size = " << _block_param.double_diff_size();
}

void Caffe::_parse_relu_param(const caffe::ReLUParameter & _block_param, const caffe::V1LayerParameter &layer)
{
	LOG_IF(INFO, _verbose) << "  -> ReLUParameter";
	LOG_IF(INFO, _verbose) << "     .negative_slope = " << _block_param.negative_slope() << " [default: 0]";
	df->leaky_relu(layer.bottom(0) + "_output_0", _block_param.negative_slope(), layer.name(), {});
}

void Caffe::_parse_dropout_param(const caffe::DropoutParameter & _block_param, const caffe::V1LayerParameter &layer)
{
	LOG_IF(INFO, _verbose) << "  -> DropoutParameter";
	LOG_IF(INFO, _verbose) << "     .dropout_ratio = " << _block_param.dropout_ratio() << " [default: 0.5]";
	df->dropout(layer.bottom(0) + "_output_0", _block_param.dropout_ratio(), true, layer.name(), {});
}

void Caffe::_parse_inner_product_param(const caffe::InnerProductParameter & _block_param, const caffe::V1LayerParameter &layer)
{
	LOG_IF(INFO, _verbose) << "  -> InnerProductParameter";
	LOG_IF(INFO, _verbose) << "     .num_output = " << _block_param.num_output();
	LOG_IF(INFO, _verbose) << "     .bias_term = " << _block_param.bias_term() << " [default: true]";
	if (_block_param.has_bias_filler()) {
		LOG_IF(INFO, _verbose) << "      .bias_filler = true";
		_parse_filler_param({}, _block_param.bias_filler(), "bias_filler");
	}
	LOG_IF(INFO, _verbose) << "     .axis = " << _block_param.axis() << " [default: 1]";
	LOG_IF(INFO, _verbose) << "     .transpose = " << _block_param.transpose() << " [default: false]";

	int num_output = _block_param.num_output();
	LOG_IF(FATAL, num_output < 1) << "num_output < 1";
	int num_inputs = layer.blobs(0).data_size() / num_output;
	//LOG_IF(FATAL, param.has_weight_filler() == false) << "param.has_weight_filler() == false - " << layer.name() ;
	std::string weight = df->variable( _parse_filler_param({ num_inputs, num_output, 1, 1 }, _block_param.weight_filler(), "weight filler"), 0, layer.name() + "_w", {});
	auto weights_node = df->block()->find_node_by_output__name(weight);
	auto weight_mutable_weights = weights_node->mutable_variable_param()->mutable_weights();
	for (auto d : layer.blobs(0).data())
		weight_mutable_weights->add_weight(d);
	if (_block_param.bias_term() == false) {
		df->matmul(layer.bottom(0) + "_output_0", weight, layer.name(), {});
	}
	else {
		//LOG_IF(FATAL, param.has_bias_filler() == false) << "param.has_bias_filler() == false - " << layer.name();
		std::string bias = df->variable(_parse_filler_param({ 1, num_output, 1, 1 }, _block_param.bias_filler(), "bias filler"), 0, layer.name() + "_b", {});
		auto bias_node = df->block()->find_node_by_output__name(bias);		
		auto bias_mutable_weights = bias_node->mutable_variable_param()->mutable_weights();
		for (auto d : layer.blobs(1).data())
			bias_mutable_weights->add_weight(d);		
		std::string m = df->matmul(layer.bottom(0) + "_output_0", weight, layer.name() + "_ip", {});
		df->bias_add(m, bias, layer.name());
	}
}

void Caffe::_parse_conv_param(const caffe::ConvolutionParameter & _block_param, const caffe::V1LayerParameter &layer)
{
	LOG_IF(INFO, _verbose) << "  -> ConvolutionParam";	

	LOG_IF(INFO, _verbose) << "     .axis = " << _block_param.axis() << " [default: 1]";
	LOG_IF(FATAL, _block_param.axis() != 1) << "Unsupported axis";

	LOG_IF(INFO, _verbose) << "     .kernel_size_size = " << _block_param.kernel_size_size();
	LOG_IF(FATAL, _block_param.kernel_size_size() > 1) << "Unsupported param.kernel_size_size() > 1";
	for (auto d : _block_param.kernel_size())
		LOG_IF(INFO, _verbose) << "      .kernel_size = " << d;
	LOG_IF(INFO, _verbose) << "     .kernel_h = " << _block_param.kernel_h();
	LOG_IF(INFO, _verbose) << "     .kernel_w = " << _block_param.kernel_w();
	int kernel_h, kernel_w;
	if (_block_param.kernel_size_size() == 1) {
		kernel_h = _block_param.kernel_size(0);
		kernel_w = kernel_h;
	}
	else {
		LOG_IF(FATAL, _block_param.kernel_h() == 0) << "No kernel_size or kernel_h is defined.";
		LOG_IF(FATAL, _block_param.kernel_w() == 0) << "No kernel_size or kernel_w is defined.";
		kernel_h = _block_param.kernel_h();
		kernel_w = _block_param.kernel_w();
	}
	LOG_IF(INFO, _verbose) << "     .stride_size = " << _block_param.stride_size() << " [default: 1]";
	for (auto d : _block_param.stride())
		LOG_IF(INFO, _verbose) << "      .stride = " << d;
	LOG_IF(INFO, _verbose) << "     .stride_h = " << _block_param.stride_h();
	LOG_IF(INFO, _verbose) << "     .stride_w = " << _block_param.stride_w();
	int stride_h, stride_w;
	if (_block_param.stride_size() == 1) {
		stride_h = _block_param.stride(0);
		stride_w = stride_h;
	}
	else {
		stride_h = _block_param.stride_h();
		stride_w = _block_param.stride_w();
	}

	if (stride_h == 0)
		stride_h = 1;
	if (stride_w == 0)
		stride_w = 1;

	LOG_IF(INFO, _verbose) << "     .pad_size = " << _block_param.pad_size() << " [default: 0]";
	for (auto d : _block_param.pad())
		LOG_IF(INFO, _verbose) << "      .pad = " << d;
	LOG_IF(INFO, _verbose) << "     .pad_h = " << _block_param.pad_h() << " [default: 0]";
	LOG_IF(INFO, _verbose) << "     .pad_w = " << _block_param.pad_w() << " [default: 0]";
	int pad_h, pad_w;
	if (_block_param.pad_size() == 1) {
		pad_h = _block_param.pad(0);
		pad_w = pad_h;
	}
	else {
		pad_h = _block_param.pad_h();
		pad_w = _block_param.pad_w();
	}
	
	LOG_IF(INFO, _verbose) << "     .dilation_size = " << _block_param.dilation_size() << " [default: 1]";
	for (auto d : _block_param.dilation())
		LOG_IF(INFO, _verbose) << "      .dilation = " << d;
	int dilation_h, dilation_w;
	if (_block_param.dilation_size() == 0) {
		dilation_h = dilation_w = 0;
	}
	else if (_block_param.dilation_size() == 1) {
		dilation_h = dilation_w = _block_param.dilation(0);
	}
	else if (_block_param.dilation_size() == 2) {
		dilation_h = _block_param.dilation(0);
		dilation_w = _block_param.dilation(1);
	} else {
		LOG_IF(FATAL, _block_param.dilation_size() > 2) << "Unsupported param.dilation_size = " << _block_param.dilation_size();
	}

	if (dilation_h == 0)
		dilation_h = 1;
	if (dilation_w == 0)
		dilation_w = 1;
	
	if (_block_param.has_weight_filler()) {
		LOG_IF(INFO, _verbose) << "     .weight_filler = true";		
	}
	//LOG_IF(FATAL, param.has_weight_filler() == false) << "param.has_weight_filler() == false - " << layer.name();
	LOG_IF(INFO, _verbose) << "     .num_output = " << _block_param.num_output();
	int num_output = _block_param.num_output();
	LOG_IF(FATAL, num_output < 1) << "num_output < 1";
	int filter_second_dimension = layer.blobs(0).data_size() / num_output / kernel_h / kernel_w;
	std::string filter = df->variable(_parse_filler_param({ num_output, filter_second_dimension, kernel_h, kernel_w }, _block_param.weight_filler(), "weight filler"), 0, layer.name() + "_w", {});
	auto filter_node = df->block()->find_node_by_output__name(filter);
	auto filter_mutable_weights = filter_node->mutable_variable_param()->mutable_weights();
	for (auto d : layer.blobs(0).data())
		filter_mutable_weights->add_weight(d);
	std::string bias;
	LOG_IF(INFO, _verbose) << "     .bias_term = " << _block_param.bias_term();
	if (_block_param.bias_term() == true) {
		if (_block_param.has_bias_filler()) {
			LOG_IF(INFO, _verbose) << "     .bias_filler = true";
			_parse_filler_param({}, _block_param.bias_filler(), "bias_filler");
		}
		//LOG_IF(FATAL, param.has_bias_filler() == false) << "param.has_bias_filler() == false - " << layer.name();
		bias = df->variable(_parse_filler_param({ 1, num_output, 1, 1 }, _block_param.bias_filler(), "bias filler"), 0, layer.name() + "_b", {});
		auto bias_node = df->block()->find_node_by_output__name(bias);
		auto bias_mutable_weights = bias_node->mutable_variable_param()->mutable_weights();
		for (auto d : layer.blobs(1).data())
			bias_mutable_weights->add_weight(d);
	}
	df->conv2d(layer.bottom(0) + "_output_0", filter, bias, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, layer.name(), {});
}

void Caffe::_parse_softmax_param(const caffe::SoftmaxParameter & _block_param, const caffe::V1LayerParameter & layer)
{
	LOG_IF(INFO, _verbose) << "  -> SoftmaxParameter";
}

void Caffe::_parse_pooling_param(const caffe::PoolingParameter & _block_param, const caffe::V1LayerParameter &layer)
{
	LOG_IF(INFO, _verbose) << "  -> PoolingParameter";
	LOG_IF(INFO, _verbose) << "     .pool_method = " << caffe::PoolingParameter_PoolMethod_Name(_block_param.pool()) << " [default: MAX]";
	LOG_IF(INFO, _verbose) << "     .pad = " << _block_param.pad() << " [default: 0]";
	LOG_IF(INFO, _verbose) << "     .pad_h = " << _block_param.pad_h() << " [default: 0]";
	LOG_IF(INFO, _verbose) << "     .pad_w = " << _block_param.pad_w() << " [default: 0]";
	LOG_IF(INFO, _verbose) << "     .kernel_size = " << _block_param.kernel_size() << " [default: 0]";
	LOG_IF(INFO, _verbose) << "     .kernel_h = " << _block_param.kernel_h();
	LOG_IF(INFO, _verbose) << "     .kernel_w = " << _block_param.kernel_w();
	LOG_IF(INFO, _verbose) << "     .stride = " << _block_param.stride() << " [default: 1]";
	LOG_IF(INFO, _verbose) << "     .stride_h = " << _block_param.stride_h();
	LOG_IF(INFO, _verbose) << "     .stride_w = " << _block_param.stride_w();
	LOG_IF(INFO, _verbose) << "     .global_pooling = " << _block_param.global_pooling() << " [default: false]";
	
	int windowHeight, windowWidth;
	if (_block_param.kernel_size() != 0) {
		windowHeight = _block_param.kernel_size();
		windowWidth = _block_param.kernel_size();
	}
	else {
		windowHeight = _block_param.kernel_h();
		windowWidth = _block_param.kernel_w();
	}
	int verticalPadding, horizontalPadding;
	if (_block_param.pad() != 0) {
		verticalPadding = _block_param.pad();
		horizontalPadding = _block_param.pad();
	}
	else {
		verticalPadding = _block_param.pad_h();
		horizontalPadding = _block_param.pad_w();
	}
	int verticalStride, horizontalStride;
	if (_block_param.stride() != 1) {
		verticalStride = _block_param.stride();
		horizontalStride = _block_param.stride();
	}
	else {
		verticalStride = _block_param.stride_w();
		horizontalStride = _block_param.stride_h();
	}

	df->pooling(layer.bottom(0) + "_output_0", windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride, layer.name(), {});
}

