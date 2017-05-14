#pragma once

#include "core/export.h"
#include "core/deep_flow.h"
#include "proto\caffe.pb.h"
#include <memory>

class DeepFlowDllExport Caffe {
	friend class DeepFlow;
private:	
	void _parse_net_deprecated(std::shared_ptr<caffe::NetParameter> net, std::initializer_list<std::pair<std::string, std::array<int, 4>>> inputs);
	void _parse_layer_deprecated(const caffe::V1LayerParameter &layer);	
	void _parse_pooling_param(const caffe::PoolingParameter &param, const caffe::V1LayerParameter &layer);
	void _parse_dropout_param(const caffe::DropoutParameter &param, const caffe::V1LayerParameter &layer);
	void _parse_inner_product_param(const caffe::InnerProductParameter &param, const caffe::V1LayerParameter &layer);
	std::string _parse_filler_param(std::initializer_list<int> dims, const caffe::FillerParameter &param, std::string name);
	void _parse_relu_param(const caffe::ReLUParameter &param, const caffe::V1LayerParameter &layer);
	void _parse_conv_param(const caffe::ConvolutionParameter &param, const caffe::V1LayerParameter &layer);
	void _parse_softmax_param(const caffe::SoftmaxParameter &param, const caffe::V1LayerParameter &layer);	
	void _parse_blob_param(const caffe::BlobProto &param);			
private:
	DeepFlow *df;
	bool _verbose;
public:
	Caffe(DeepFlow *df, bool verbose);
	void load(std::string file_path, std::initializer_list<std::pair<std::string,std::array<int,4>>> inputs);
};