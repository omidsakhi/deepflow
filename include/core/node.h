#pragma once

#include "core/export.h"

#include <functional>
#include <memory>
#include <vector>

#include "cudnn.h"

#include "core/tensor.h"
#include "core/cuda_helper.h"
#include "core/execution_context.h"

#include <list>

#include <initializer_list>
#include "core/terminal.h"

#include <string>
#include <vector>

#include <iostream>

#include "proto/deepflow.pb.h"

class NodeObserver;

class DeepFlowDllExport Node : public std::enable_shared_from_this<Node>, public CudaHelper {
public:
	enum TraverseOrder {
		PRE_ORDER,
		POST_ORDER
	};
	Node();
	Node(deepflow::NodeParam *param);
	void createIO();
	virtual void init() = 0;
	virtual int minNumInputs() = 0;
	virtual int minNumOutputs() = 0;
	virtual void forward() = 0;
	virtual void backward() = 0;	
	virtual bool isGenerator() { return false; }
	virtual bool isLastBatch() { return true; }
	virtual std::string to_cpp() const = 0;
	virtual void prep_for_saving() {}
	virtual void reset_gradients();
	std::string name() const;	
	std::string _to_cpp_phases() const;
	std::string _input_name_for_cpp(int i) const;	
	void write_values(std::shared_ptr<Tensor> tensor, float alpha = 1.0, float beta = 0.0f);
	void write_diffs(std::shared_ptr<Tensor> tensor, float alpha = 1.0, float beta = 0.0f);
	void cpy(int n, const float alpha, const void *src, const float beta, void *dst);
	void dot(const int n, const float alpha, const void *a, const void *b, const float beta, void *dst);
	void fill(int n, const float value, void *dst, const float beta = 0);
	void fill(const float value);
	std::vector<NodeInputPtr> &inputs();
	std::vector<NodeOutputPtr> &outputs();
	NodeInputPtr input(int index);
	NodeOutputPtr output(int index);
	bool isInitialized() const;
	void setInitialized(bool status);
	deepflow::NodeParam *param();
	bool includePhase(const std::string &phase);
	void setExecutionContext(ExecutionContextPtr context);
	ExecutionContextPtr executionContext();
	virtual std::list<std::shared_ptr<Node>> inputNodes() const;
	virtual std::list<std::shared_ptr<Node>> outputNodes() const;
protected:	
	std::vector<NodeInputPtr> _inputs;
	std::vector<NodeOutputPtr> _outputs;
	std::string _name = "Unknown";	
	bool _initialized = false;
	deepflow::NodeParam *_param = nullptr;
	ExecutionContextPtr _context = nullptr;
	const float one = 1.0f;
	const float zero = 0.0f;
	int _verbose = 0;
};

using NodePtr = std::shared_ptr<Node>;

__global__
void GrayPictureGeneratorKernel(const int num_images, const float * in, const int per_image_height, const int per_image_width, const int num_image_per_row_and_col, unsigned char * out);

__global__
void ColorPictureGeneratorKernel(const int num_images, const float * in, const int per_image_height, const int per_image_width, const int num_image_per_row_and_col, unsigned char * out);
