#pragma once

#include "core/export.h"
#include "core/deep_flow.h"
#include "proto\caffe.pb.h"
#include <memory>

class DeepFlowDllExport Caffe {
	friend class DeepFlow;
private:	
	void _parse_net_deprecated(std::shared_ptr<caffe::NetParameter> net);
	void _parse_layer_deprecated(const caffe::V1LayerParameter &layer);
	void _parse_conv_param(const caffe::ConvolutionParameter &param);	
	void _parse_filler_param(const caffe::FillerParameter &param, std::string name);
	void _parse_blob_param(const caffe::BlobProto &param);
	void _parse_relu_param(const caffe::ReLUParameter &param);
	void _parse_dropout_param(const caffe::DropoutParameter &param);
	void _parse_inner_product_param(const caffe::InnerProductParameter &param);
	void _parse_pooling_param(const caffe::PoolingParameter &param);
private:
	DeepFlow *df;
	bool _verbose;
public:
	Caffe(DeepFlow *df, bool verbose);
	void load(std::string file_path);
};