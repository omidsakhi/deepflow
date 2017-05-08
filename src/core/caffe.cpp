#include "core/caffe.h"

Caffe::Caffe(DeepFlow * df, bool verbose)
{
	this->df = df;
	_verbose = verbose;
}

void Caffe::load(std::string file_path)
{
	auto net = std::make_shared<caffe::NetParameter>();
	std::fstream input(file_path, std::ios::in | std::ios::binary);
	LOG_IF(FATAL, !net->ParseFromIstream(&input)) << "Failed to read caffe model " << file_path;
	input.close();
	LOG(INFO) << "net.layers_size = " << net->layers_size();
	LOG(INFO) << "net.layer_size = " << net->layer_size();
	if (net->layer_size() > 0) {
		LOG(INFO) << "Loading Caffe model from LayerParameter (NOT IMPLEMENTED)";
	}
	else if (net->layers_size() > 0) {
		_parse_net_deprecated(net);
	}
}

void Caffe::_parse_net_deprecated(std::shared_ptr<caffe::NetParameter> net)
{
	LOG_IF(INFO, _verbose) << "Loading Caffe model from V1LayerParameter (Deprecated) ...";
	LOG_IF(INFO, _verbose) << "net.input_shape_size = " << net->input_shape_size();	
	LOG_IF(INFO, _verbose) << "net.input_dim_size = " << net->input_dim_size();	
	LOG_IF(INFO, _verbose) << "net.input_shape_size = " << net->input_shape_size();
	LOG_IF(INFO, _verbose) << "net.input_size = " << net->input_size();	
	for (auto i : net->input()) {
		LOG_IF(INFO, _verbose) << "    input = " << i;
	}
	LOG_IF(INFO, _verbose) << "net.layers_size = " << net->layers_size();
	for (auto layer : net->layers())
		_parse_layer_deprecated(layer);	
}

void Caffe::_parse_layer_deprecated(const caffe::V1LayerParameter & layer)
{
	LOG_IF(INFO, _verbose) << "Type = " << caffe::V1LayerParameter_LayerType_Name(layer.type());
	LOG_IF(INFO, _verbose) << " .name = " << layer.name();	
	LOG_IF(INFO, _verbose) << " .bottom_size = " << layer.bottom_size();
	for (auto b : layer.bottom())
		LOG_IF(INFO, _verbose) << "    .bottom = " << b;
	LOG_IF(INFO, _verbose) << " .top_size = " << layer.top_size();
	for (auto t : layer.top())
		LOG_IF(INFO, _verbose) << "    .top = " << t;
	if (layer.has_convolution_param()) {
		_parse_conv_param(layer.convolution_param());
	}	
	if (layer.has_relu_param()) {
		_parse_relu_param(layer.relu_param());
	}	
	if (layer.has_pooling_param()) {
		_parse_pooling_param(layer.pooling_param());
	}
	if (layer.has_data_param()) {
		LOG_IF(INFO, _verbose) << " .data_param = true";
	}
	if (layer.has_inner_product_param()) {
		_parse_inner_product_param(layer.inner_product_param());
	}
	if (layer.blobs_size() > 0) {
		LOG_IF(INFO, _verbose) << " .blob_size = " << layer.blobs_size();
		for (auto blob : layer.blobs())
			_parse_blob_param(blob);
	}
	if (layer.has_dropout_param()) {
		_parse_dropout_param(layer.dropout_param());
	}
	if (layer.include_size() > 0) {
		LOG_IF(INFO, _verbose) << " .include_size = " << layer.include_size();
	}	
}

void Caffe::_parse_conv_param(const caffe::ConvolutionParameter & param)
{
	LOG_IF(INFO, _verbose) << "  -> ConvolutionParam";
	LOG_IF(INFO, _verbose) << "     .num_output = " << param.num_output();
	LOG_IF(INFO, _verbose) << "     .axis = " << param.axis() << " [default: 1]";
	LOG_IF(INFO, _verbose) << "     .bias_term = " << param.bias_term();
	LOG_IF(INFO, _verbose) << "     .dilation_size = " << param.dilation_size() << " [default: 1]";
	for (auto d : param.dilation())
		LOG_IF(INFO, _verbose) << "      .dilation = " << d;
	LOG_IF(INFO, _verbose) << "     .pad_size = " << param.pad_size() << " [default: 0]";
	for (auto d : param.pad())
		LOG_IF(INFO, _verbose) << "      .pad = " << d;
	LOG_IF(INFO, _verbose) << "     .kernel_size_size = " << param.kernel_size_size();
	for (auto d : param.kernel_size())
		LOG_IF(INFO, _verbose) << "      .kernel_size = " << d;
	LOG_IF(INFO, _verbose) << "     .stride_size = " << param.stride_size() << " [default: 1]";
	for (auto d : param.stride())
		LOG_IF(INFO, _verbose) << "      .stride = " << d;
	LOG_IF(INFO, _verbose) << "     .pad_h = " << param.pad_h() << " [default: 0]";
	LOG_IF(INFO, _verbose) << "     .pad_w = " << param.pad_w() << " [default: 0]";
	LOG_IF(INFO, _verbose) << "     .kernel_h = " << param.kernel_h();
	LOG_IF(INFO, _verbose) << "     .kernel_w = " << param.kernel_w();
	LOG_IF(INFO, _verbose) << "     .stride_h = " << param.stride_h();
	LOG_IF(INFO, _verbose) << "     .stride_w = " << param.stride_w();
	if (param.has_weight_filler()) {
		LOG_IF(INFO, _verbose) << "     .weight_filler = true";
		_parse_filler_param(param.weight_filler(), "weight_filler");
	}
	if (param.has_bias_filler()) {
		LOG_IF(INFO, _verbose) << "     .bias_filler = true";
		_parse_filler_param(param.bias_filler(), "bias_filler");
	}
}

void Caffe::_parse_filler_param(const caffe::FillerParameter & param, std::string name)
{
	// LOG_IF(INFO, _verbose) << " . = " << param. << " [default: ]";
	LOG_IF(INFO, _verbose) << "  -> FillerParameter " << name;
	LOG_IF(INFO, _verbose) << "     .type = " << param.type() << " [default: constant]";
	LOG_IF(INFO, _verbose) << "     .value = " << param.value() << " [default: 0]"; // constant
	LOG_IF(INFO, _verbose) << "     .min = " << param.min() << " [default: 0]"; // uniform
	LOG_IF(INFO, _verbose) << "     .max = " << param.max() << " [default: 1]"; // uniform
	LOG_IF(INFO, _verbose) << "     .mean = " << param.mean() << " [default: 0]"; // normal
	LOG_IF(INFO, _verbose) << "     .std = " << param.std() << " [default: 1]"; // normal
	LOG_IF(INFO, _verbose) << "     .sparse = " << param.sparse() << " [default: -1]";
	LOG_IF(INFO, _verbose) << "     .variance_norm = " << caffe::FillerParameter_VarianceNorm_Name(param.variance_norm()) << " [default: FAN_IN]"; // xavier and msra
}

void Caffe::_parse_blob_param(const caffe::BlobProto & param)
{
	LOG_IF(INFO, _verbose) << "  -> BlobProto";
	LOG_IF(INFO, _verbose) << "     .data_size = " << param.data_size();
	LOG_IF(INFO, _verbose) << "     .diff_size = " << param.diff_size();	
	LOG_IF(INFO, _verbose) << "     .double_data_size = " << param.double_data_size();
	LOG_IF(INFO, _verbose) << "     .double_diff_size = " << param.double_diff_size();
}

void Caffe::_parse_relu_param(const caffe::ReLUParameter & param)
{
	LOG_IF(INFO, _verbose) << "  -> ReLUParameter";
	LOG_IF(INFO, _verbose) << "     .negative_slope = " << param.negative_slope() << " [default: 0]";
}

void Caffe::_parse_dropout_param(const caffe::DropoutParameter & param)
{
	LOG_IF(INFO, _verbose) << "  -> DropoutParameter";
	LOG_IF(INFO, _verbose) << "     .dropout_ratio = " << param.dropout_ratio() << " [default: 0.5]";
}

void Caffe::_parse_inner_product_param(const caffe::InnerProductParameter & param)
{
	LOG_IF(INFO, _verbose) << "  -> InnerProductParameter";
	LOG_IF(INFO, _verbose) << "     .num_output = " << param.num_output();
	LOG_IF(INFO, _verbose) << "     .bias_term = " << param.bias_term() << " [default: true]";
	if (param.has_weight_filler()) {
		LOG_IF(INFO, _verbose) << "      .weight_filler = true";
		_parse_filler_param(param.weight_filler(), "weight_filler");
	}
	if (param.has_bias_filler()) {
		LOG_IF(INFO, _verbose) << "      .bias_filler = true";
		_parse_filler_param(param.bias_filler(), "bias_filler");
	}
	LOG_IF(INFO, _verbose) << "     .axis = " << param.axis() << " [default: 1]";
	LOG_IF(INFO, _verbose) << "     .transpose = " << param.transpose() << " [default: false]";
}

void Caffe::_parse_pooling_param(const caffe::PoolingParameter & param)
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
}

