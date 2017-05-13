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
	void _parse_pooling_param(const caffe::PoolingParameter &_block_param, const caffe::V1LayerParameter &layer);
	void _parse_dropout_param(const caffe::DropoutParameter &_block_param, const caffe::V1LayerParameter &layer);
	void _parse_inner_product_param(const caffe::InnerProductParameter &_block_param, const caffe::V1LayerParameter &layer);
	std::shared_ptr<deepflow::InitParam> _parse_filler_param(std::initializer_list<int> dims, const caffe::FillerParameter &_block_param, std::string name);
	void _parse_relu_param(const caffe::ReLUParameter &_block_param, const caffe::V1LayerParameter &layer);
	void _parse_conv_param(const caffe::ConvolutionParameter &_block_param, const caffe::V1LayerParameter &layer);
	void _parse_softmax_param(const caffe::SoftmaxParameter &_block_param, const caffe::V1LayerParameter &layer);	
	void _parse_blob_param(const caffe::BlobProto &_block_param);			
private:
	DeepFlow *df;
	bool _verbose;
public:
	Caffe(DeepFlow *df, bool verbose);
	void load(std::string file_path, std::initializer_list<std::pair<std::string,std::array<int,4>>> inputs);
};