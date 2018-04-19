#pragma once

#include "core/export.h"

#include <string>
#include <list>

template<class T>
class DeepFlowDllExport NodeOp {
public:
	std::string _name = "Node";
	std::string _scope = "Default";
public:
	T &name(std::string name) {
		this->_name = name;
		return *((T*)this);
	}
	T &scope(std::string scope) {
		this->_scope = scope;
		return *((T*)this);		
	}
};

class DeepFlowDllExport MNISTReaderOp : public NodeOp<MNISTReaderOp> {
public:
	enum MNISTDataset {
		TRAIN,
		TEST
	};

	enum MNISTOutput {
		DATA,
		LABELS
	};
	MNISTDataset _dataset = TRAIN;
	MNISTOutput _output = DATA;
	int _batch_size = 100;
public:
	MNISTReaderOp(std::string name = "mnist") {
		this->name(name);		
	}
	MNISTReaderOp &train() {
		this->_dataset = TRAIN;
		return *this;
	}
	MNISTReaderOp &test() {
		this->_dataset = TEST;
		return *this;
	}
	MNISTReaderOp &data() {
		this->_output = DATA;
		return *this;
	}
	MNISTReaderOp &labels() {
		this->_output = LABELS;
		return *this;
	}
	MNISTReaderOp &batch(int size) {
		this->_batch_size = size;
		return *this;
	}
};

class DeepFlowDllExport DataGeneratorOp : public NodeOp<DataGeneratorOp> {
public:
	std::string _solver;
public:
	DataGeneratorOp(const std::string &name = "data_generator") {
		this->name(name);
	}
	DataGeneratorOp &solver(std::string solver) {
		this->_solver = solver;
		return *this;
	}
};

class DeepFlowDllExport TextImageGeneratorOp : public NodeOp<TextImageGeneratorOp> {
public:
	std::string _chars = "AaBbCcDdEeFfGgHhIiJjKkLlMmOoPpQqRrSsTtUuVvWwXxYyZz0123456789%.";
	std::list<std::string> _words;
public:
	TextImageGeneratorOp(std::string name = "text_image_generator") {
		this->name(name);
	}
	TextImageGeneratorOp &no_chars() {
		this->_chars = "";
		return *this;
	}
	TextImageGeneratorOp &chars(std::string chars) {
		this->_chars = chars;
		return *this;
	}
	TextImageGeneratorOp &words(std::list<std::string> words) {
		this->_words = words;
		return *this;
	}
};

class DeepFlowDllExport ImreadOp : public NodeOp<ImreadOp> {
public:
	enum ImreadImageType {
		IMREAD_GRAY,
		IMREAD_COLOR
	};	
	ImreadImageType _type = IMREAD_COLOR;
public:
	ImreadOp(std::string name = "imread") {		
		this->name(name);
	}
	ImreadOp &color() {
		this->_type = IMREAD_COLOR;
		return *this;
	}
	ImreadOp &gray() {
		this->_type = IMREAD_GRAY;
		return *this;
	}
};

class DeepFlowDllExport ImbatchOp : public NodeOp<ImbatchOp> {
public:
	bool _randomize = true;
	bool _between_0_and_1 = false;
public:
	ImbatchOp(std::string name = "imbatch") {
		this->name(name);
	}
	ImbatchOp &randomize(bool state) {
		this->_randomize = state;
		return *this;
	}
	ImbatchOp &between_0_and_1() {
		this->_between_0_and_1 = true;
		return *this;
	}
};

class VariableOp : public NodeOp<VariableOp> {
public:
	VariableOp(std::string name = "var") {
		this->name(name);
	}
};

class PlaceholderOp : public NodeOp<PlaceholderOp> {
public:
	enum PlaceholderDataType {
		FLOAT = 0,
		DOUBLE = 1,
		HALF = 2,
		INT8 = 3,
		INT32 = 4,
		INT8x4 = 5,
	};
	PlaceholderDataType _type = FLOAT;
public:
	PlaceholderOp(std::string name = "placeholder") {
		this->name(name);
	}
	PlaceholderOp &floatType() {
		this->_type = FLOAT;
		return *this;
	}
	PlaceholderOp &intType() {
		this->_type = INT32;
		return *this;
	}
};

class ConvolutionOp : public NodeOp<ConvolutionOp> {
public:
	int _pad_h = 1;
	int _pad_w = 1;	
	int _kernel_h = 3;
	int _kernel_w = 3;
	int _stride_u = 1;
	int _stride_v = 1;
	int _dilation_h = 1;
	int _dilation_w = 1;
	bool _with_bias = false;
	float _stddev = 0.02f;
	bool _same = true;
public:
	ConvolutionOp(std::string name = "conv") {
		this->name(name);
	}
	ConvolutionOp &pad_w(int size) {
		this->_pad_w = size;
		this->_same = false;
		return *this;
	}
	ConvolutionOp &pad_h(int size) {
		this->_pad_h = size;
		this->_same = false;
		return *this;
	}	
	ConvolutionOp &pad(int size) {
		this->_pad_h = size;
		this->_pad_w = size;
		this->_same = false;
		return *this;
	}
	ConvolutionOp &same() {		
		this->_same = true;
		_pad_h = floor(_kernel_h / 2);
		_pad_w = floor(_kernel_w / 2);
		return *this;
	}
	ConvolutionOp &kernel_h(int size) {
		this->_kernel_h = size;
		if (_same) {
			_pad_h = floor(_kernel_h / 2);
		}
		return *this;
	}
	ConvolutionOp &kernel_w(int size) {
		this->_kernel_w = size;
		if (_same) {
			_pad_w = floor(_kernel_w / 2);
		}
		return *this;
	}
	ConvolutionOp &kernel(int size) {
		this->_kernel_h = size;
		this->_kernel_w = size;
		if (_same) {
			_pad_h = floor(_kernel_h / 2);
			_pad_w = floor(_kernel_w / 2);
		}
		return *this;
	}
	ConvolutionOp &stride_u(int size) {
		this->_stride_u = size;
		return *this;
	}
	ConvolutionOp &stride_v(int size) {
		this->_stride_v = size;
		return *this;
	}
	ConvolutionOp &stride(int size) {
		this->_stride_u = size;
		this->_stride_v = size;
		return *this;
	}
	ConvolutionOp &dilation_h(int size) {
		this->_dilation_h = size;
		return *this;
	}
	ConvolutionOp &dilation_w(int size) {
		this->_dilation_w = size;
		return *this;
	}
	ConvolutionOp &dilation(int size) {
		this->_dilation_h = size;
		this->_dilation_w = size;
		return *this;
	}
	ConvolutionOp &with_bias() {
		this->_with_bias = true;
		return *this;
	}
	ConvolutionOp &stddev(float value) {
		this->_stddev = value;
		return *this;
	}
};

class RestructureOp : public NodeOp<RestructureOp> {
public:
	RestructureOp(std::string name = "restructure") {
		this->name(name);
	}
};

class PoolingOp : public NodeOp<PoolingOp> {
public:
	int _window_h = 2;
	int _window_w = 2;
	int _v_pad = 0;
	int _h_pad = 0;
	int _v_stride = 1;
	int _h_stride = 1;
	bool _same = false;
public:
	PoolingOp(std::string name = "pooling") {
		this->name(name);
	}
	PoolingOp &same() {
		this->_same = true;
		_v_pad = floor(_window_h / 2);
		_h_pad = floor(_window_w / 2);
		return *this;
	}
	PoolingOp &v_pad(int size) {
		this->_v_pad = size;
		this->_same = false;
		return *this;
	}
	PoolingOp &h_pad(int size) {
		this->_h_pad = size;
		this->_same = false;
		return *this;
	}
	PoolingOp &pad(int size) {
		this->_v_pad = size;
		this->_h_pad = size;
		this->_same = false;
		return *this;
	}
	PoolingOp &window_h(int size) {
		this->_window_h = size;
		if (_same) {
			_v_pad = floor(_window_h / 2);
		}
		return *this;
	}
	PoolingOp &window_w(int size) {
		this->_window_w = size;
		if (_same) {
			_h_pad = floor(_window_w / 2);
		}
		return *this;
	}
	PoolingOp &window(int size) {
		this->_window_h = size;
		this->_window_w = size;
		if (_same) {
			_h_pad = floor(_window_w / 2);
			_v_pad = floor(_window_h / 2);
		}
		return *this;
	}
	PoolingOp &v_stride(int size) {
		this->_v_stride = size;
		return *this;
	}
	PoolingOp &h_stride(int size) {
		this->_h_stride = size;
		return *this;
	}
	PoolingOp &stride(int size) {
		this->_v_stride = size;
		this->_h_stride = size;
		return *this;
	}
};

class LiftingOp : public NodeOp<LiftingOp> {
public:
	bool _with_flip = false;
	bool _up = true;	
public:
	LiftingOp(std::string name = "lift") {
		this->name(name);
	}
	LiftingOp &with_flip() {
		_with_flip = true;		
		return *this;
	}
	LiftingOp &up() {
		_up = true;
		return *this;
	}
	LiftingOp &down() {
		_up = false;
		return *this;
	}
};

class PatchingOp : public NodeOp<PatchingOp> {
public:
	bool _samples = false;  // samples vs channels
	bool _up = true; // up vs down
	int _h_num = 2;
	int _v_num = 2;
public:
	PatchingOp(std::string name = "patching") {
		this->name(name);
	}
	PatchingOp &samples() {
		_samples = true;
		return *this;
	}
	PatchingOp &channels() {
		_samples = false;
		return *this;
	}
	PatchingOp &up() {
		_up = true;
		return *this;
	}
	PatchingOp &down() {
		_up = false;
		return *this;
	}
	PatchingOp &h(int count) {
		_h_num = count;
		return *this;
	}
	PatchingOp &v(int count) {
		_v_num = count;
		return *this;
	}
};

class PatchSamplingOp : public NodeOp<PatchSamplingOp> {
public:
	PatchSamplingOp(std::string name = "patch_sampling") {
		this->name(name);
	}
};

class ResizeOp : public NodeOp<ResizeOp> {
public:
	ResizeOp(std::string name = "resize") {
		this->name(name);
	}
};

class ReshapeOp : public NodeOp<ReshapeOp> {
public:
	ReshapeOp(std::string name = "reshape") {
		this->name(name);
	}
};

class ConcateOp : public NodeOp<ConcateOp> {
public:
	ConcateOp(std::string name = "concate") {
		this->name(name);
	}
};

class AddOp : public NodeOp<AddOp> {
public:
	float _alpha = 1.0;
	float _beta = 1.0;
public:
	AddOp(std::string name = "add") {
		this->name(name);
	}
	AddOp &alpha(float coef) {
		_alpha = coef;
		return *this;
	}
	AddOp &beta(float coef) {
		_beta = coef;
		return *this;
	}
};

class SubtractOp : public NodeOp<SubtractOp> {
public:
	SubtractOp(std::string name = "subtract") {
		this->name(name);
	}
};

class DotOp : public NodeOp<DotOp> {
public:
	DotOp(std::string name = "dot") {
		this->name(name);
	}
};

class SquareOp : public NodeOp<SquareOp> {
public:
	SquareOp(std::string name = "square") {
		this->name(name);
	}
};

class AbsOp : public NodeOp<AbsOp> {
public:
	AbsOp(std::string name = "abs") {
		this->name(name);
	}
};

class ExpOp : public NodeOp<ExpOp> {
public:
	ExpOp(std::string name = "exp") {
		this->name(name);
	}
};

class LogOp : public NodeOp<LogOp> {
public:
	float _coef = 1.0;
public:
	LogOp(std::string name = "log") {
		this->name(name);
	}
	LogOp &coef(float value) {
		_coef = value;
		return *this;
	}
	LogOp &negate() {
		_coef = -1.0f;
		return *this;
	}
};

class SquareErrorOp : public NodeOp<SquareErrorOp> {
public:
	SquareErrorOp(std::string name = "square_error") {
		this->name(name);
	}
};

class SoftmaxOp : public NodeOp<SoftmaxOp> {
public:
	enum SoftmaxMode {
		INSTANCE,
		CHANNEL
	};
	SoftmaxMode _mode;
public:
	SoftmaxOp(std::string name = "softmax") {
		this->name(name);
	}
	SoftmaxOp &by_instance() {
		_mode = INSTANCE;
		return *this;
	}
	SoftmaxOp &by_channel() {
		_mode = CHANNEL;
		return *this;
	}
};

class MaxOp : public NodeOp<MaxOp> {
public:
	MaxOp(std::string name = "max") {
		this->name(name);
	}
};

class MatmulOp : public NodeOp<MatmulOp> {
public:
	MatmulOp(std::string name = "matmul") {
		this->name(name);
	}
};

class BiasAddOp : public NodeOp<BiasAddOp> {
public:
	BiasAddOp(std::string name = "bias_add") {
		this->name(name);
	}
};

class DenseOp : public NodeOp<DenseOp> {
public:
	bool _with_bias = true;
	float _stddev = 0.02f;
public:
	DenseOp(std::string name = "dense") {
		this->name(name);
	}
	DenseOp &with_bias() {
		this->_with_bias = true;
		return *this;
	}
	DenseOp &no_bias() {
		this->_with_bias = false;
		return *this;
	}
	DenseOp &stddev(float value) {
		this->_stddev = value;
		return *this;
	}
};

class DropoutOp : public NodeOp<DropoutOp> {
public:
	float _ratio = 0.5f;
public:
	DropoutOp(std::string name = "dropout") {
		this->name(name);
	}
	DropoutOp &ratio(float value) {
		this->_ratio = value;
		return *this;
	}
};

class ReduceOp : public NodeOp<ReduceOp> {
public:
	int _reduce_dimention = 1;
	deepflow::ReduceParam_ReduceOp _op = deepflow::ReduceParam_ReduceOp_AVG;
	deepflow::ReduceParam::OutputType _output_type = deepflow::ReduceParam_OutputType_VALUES;
	int _terminal = 0;
public:
	ReduceOp(std::string name = "reduce") {
		this->name(name);
	}
	ReduceOp &reduce(int dimension) {
		this->_reduce_dimention = dimension;
		return *this;
	}
	ReduceOp &avg() {
		this->_op = deepflow::ReduceParam_ReduceOp_AVG;
		return *this;
	}
	ReduceOp &add() {
		this->_op = deepflow::ReduceParam_ReduceOp_ADD;
		return *this;
	}
	ReduceOp &mul() {
		this->_op = deepflow::ReduceParam_ReduceOp_MUL;
		return *this;
	}
	ReduceOp &max() {
		this->_op = deepflow::ReduceParam_ReduceOp_MAX;
		return *this;
	}
	ReduceOp &min() {
		this->_op = deepflow::ReduceParam_ReduceOp_MIN;
		return *this;
	}
	ReduceOp &amax() {
		this->_op = deepflow::ReduceParam_ReduceOp_AMAX;
		return *this;
	}
	ReduceOp &norm1() {
		this->_op = deepflow::ReduceParam_ReduceOp_NORM1;
		return *this;
	}
	ReduceOp &norm2() {
		this->_op = deepflow::ReduceParam_ReduceOp_NORM2;
		return *this;
	}
	ReduceOp &values() {
		this->_output_type = deepflow::ReduceParam_OutputType_VALUES;
		return *this;
	}
	ReduceOp &indicies() {
		this->_output_type = deepflow::ReduceParam_OutputType_INDICES;
		return *this;
	}
	ReduceOp &terminal(int num) {
		this->_terminal = num;
		return *this;
	}
};

class ArgmaxOp : public NodeOp<ArgmaxOp> {
public:
	ArgmaxOp(std::string name = "argmax") {
		this->name(name);
	}
};

class ArgminOp : public NodeOp<ArgminOp> {
public:
	ArgminOp(std::string name = "argmin") {
		this->name(name);
	}
};

class ReduceMaxOp : public NodeOp<ReduceMaxOp> {
public:
	ReduceMaxOp(std::string name = "reduce_max") {
		this->name(name);
	}
};

class ReduceMinOp : public NodeOp<ReduceMinOp> {
public:
	ReduceMinOp(std::string name = "reduce_min") {
		this->name(name);
	}
};

class ReduceMeanOp : public NodeOp<ReduceMeanOp> {
public:
	ReduceMeanOp(std::string name = "reduce_mean") {
		this->name(name);
	}
};

class ReduceSumOp : public NodeOp<ReduceSumOp> {
public:
	ReduceSumOp(std::string name = "reduce_sum") {
		this->name(name);
	}
};

class ReduceAmaxOp : public NodeOp<ReduceAmaxOp> {
public:
	ReduceAmaxOp(std::string name = "reduce_amax") {
		this->name(name);
	}
};

class ReduceNorm1Op : public NodeOp<ReduceNorm1Op> {
public:
	ReduceNorm1Op(std::string name = "reduce_norm1") {
		this->name(name);
	}
};

class ReduceNorm2Op : public NodeOp<ReduceNorm2Op> {
public:
	ReduceNorm2Op(std::string name = "reduce_norm2") {
		this->name(name);
	}
};

class ReduceAllOp : public NodeOp<ReduceAllOp> {
public:
	enum ReduceAllType {
		REDUCE_ALL_SUM,
		REDUCE_ALL_AVG
	};
	ReduceAllType _op = REDUCE_ALL_AVG;
public:
	ReduceAllOp(std::string name = "reduce_all") {
		this->name(name);
	}
	ReduceAllOp &sum() {
		this->_op = REDUCE_ALL_SUM;
		return *this;
	}
	ReduceAllOp &mean() {
		this->_op = REDUCE_ALL_AVG;
		return *this;
	}
};

class LeakyReluOp : public NodeOp<LeakyReluOp> {
public:
	float _negative_slope = 0.2f;
public:
	LeakyReluOp(std::string name = "leaky_relu") {
		this->name(name);
	}
	LeakyReluOp &negative_slope(float value) {
		this->_negative_slope = value;
		return *this;
	}
};

class SigmoidOp : public NodeOp<SigmoidOp> {
public:
	SigmoidOp(std::string name = "sigmoid") {
		this->name(name);
	}
};

class TanhOp : public NodeOp<TanhOp> {
public:
	TanhOp(std::string name = "tanh") {
		this->name(name);
	}
};

class ReluOp : public NodeOp<ReluOp> {
public:
	ReluOp(std::string name = "relu") {
		this->name(name);
	}
};

class ClippedReluOp : public NodeOp<ClippedReluOp> {
public:
	float _threshold = 0;
public:
	ClippedReluOp(std::string name = "clipped_relu") {
		this->name(name);
	}
	ClippedReluOp &threshold(float value) {
		this->_threshold = value;
		return *this;
	}
};

class EluOp : public NodeOp<EluOp> {
public:
	float _alpha = 0.2f;
public:
	EluOp(std::string name = "elu") {
		this->name(name);
	}
	EluOp &alpha(float coef) {
		this->_alpha = coef;
		return *this;
	}
};

class PReluOp : public NodeOp<PReluOp> {
public:
	float _initial_slope = 0.2f;
public:
	PReluOp(std::string name = "prelu") {
		this->name(name);
	}
	PReluOp &initial_slope(float value) {
		this->_initial_slope = value;
		return *this;
	}
};

class DPReluOp : public NodeOp<DPReluOp> {
public:
	float _negative_slope = 0.2f;
public:
	DPReluOp(std::string name = "dprelu") {
		this->name(name);
	}
	DPReluOp &negative_slope(float value) {
		this->_negative_slope = value;
		return *this;
	}
};

class LossOp : public NodeOp<LossOp> {
public:
	enum LossType {
		SUM,
		AVG
	};
	float _alpha = 1.0f;
	float _beta = 0.0f;
	LossType _type = AVG;
public:
	LossOp(std::string name = "loss") {
		this->name(name);
	}
	LossOp &alpha(float value) {
		this->_alpha = value;
		return *this;
	}
	LossOp &beta(float value) {
		this->_beta = value;
		return *this;
	}
	LossOp &avg() {
		this->_type = AVG;
		return *this;
	}
	LossOp &sum() {
		this->_type = SUM;
		return *this;
	}
};

class RandomSelectorOp : public NodeOp<RandomSelectorOp> {
public:
	float _ratio = 0.5f;
public:
	RandomSelectorOp(std::string name = "selector") {
		this->name(name);
	}
	RandomSelectorOp &ratio(float value) {
		this->_ratio = value;
		return *this;
	}
};

class MultiplexerOp : public NodeOp<MultiplexerOp> {
public:
	MultiplexerOp(std::string name = "multiplexer") {
		this->name(name);
	}
};

class SwitchOp : public NodeOp<SwitchOp> {
public:
	SwitchOp(std::string name = "switch") {
		this->name(name);
	}
};

class PrintOp : public NodeOp<PrintOp> {
public:
	bool _values = true;
	bool _diffs = false;
public:
	PrintOp(std::string name = "print") {
		this->name(name);
	}
	PrintOp &values() {
		this->_values = true;
		return *this;
	}
	PrintOp &diffs() {
		this->_diffs = true;
		return *this;
	}
	PrintOp &both() {
		this->_values = true;
		this->_diffs = true;
		return *this;
	}
};

class LoggerOp : public NodeOp<LoggerOp> {
public:
	bool _values = true;
	bool _diffs = false;
public:
	LoggerOp(std::string name = "logger") {
		this->name(name);
	}
	LoggerOp &values() {
		this->_values = true;
		return *this;
	}
	LoggerOp &diffs() {
		this->_diffs = true;
		return *this;
	}
	LoggerOp &both() {
		this->_values = true;
		this->_diffs = true;
		return *this;
	}
};

class DisplayOp : public NodeOp<DisplayOp> {
public:
	bool _values = true;
	bool _diffs = false;
	int _iter_frequency = 1;
	int _epoch_freuqency = 1;
	int _delay_msec = 1;
	bool _draw_iterations = true;
public:
	DisplayOp(std::string name = "disp") {
		this->name(name);
	}
	DisplayOp &values() {
		this->_values = true;
		this->_diffs = false;
		return *this;
	}
	DisplayOp &diffs() {
		this->_diffs = true;
		this->_values = false;
		return *this;
	}
	DisplayOp &iter(int disp_frequency) {
		this->_iter_frequency = disp_frequency;
		return *this;
	}
	DisplayOp &epoch(int disp_frequency) {
		this->_epoch_freuqency = disp_frequency;
		return *this;
	}
	DisplayOp &delay(int msec) {
		this->_delay_msec = msec;
		return *this;
	}
	DisplayOp &draw() {
		this->_draw_iterations = true;
		return *this;
	}
	DisplayOp &no_draw() {
		this->_draw_iterations = false;
		return *this;
	}
};

class ImwriteOp : public NodeOp<ImwriteOp> {
public:
	ImwriteOp(std::string name = "imwrite") {
		this->name(name);
	}
};

class PsnrOp : public NodeOp<PsnrOp> {
public:
	PsnrOp(std::string name = "psnr") {
		this->name(name);
	}
};

class BatchNormalizationOp : public NodeOp<BatchNormalizationOp> {
public:
	enum NormalizationMode {
		PER_ACTIVATION = 0,
		SPATIAL = 1
	};
	NormalizationMode _mode = SPATIAL;
	float _exponent_average_factor = 0.1f;
	bool _cache = true;
	float _eps = CUDNN_BN_MIN_EPSILON;
public:
	BatchNormalizationOp(std::string name = "batchnorm") {
		this->name(name);
	}
	BatchNormalizationOp &per_activation() {
		this->_mode = PER_ACTIVATION;
		return *this;
	}
	BatchNormalizationOp &spatial() {
		this->_mode = SPATIAL;
		return *this;
	}
	BatchNormalizationOp &exponent_factor(float value) {
		this->_exponent_average_factor = value;
		return *this;
	}
	BatchNormalizationOp &momentum(float value) {
		this->_exponent_average_factor = value;
		return *this;
	}
	BatchNormalizationOp &decay(float value) {
		this->_exponent_average_factor = 1 - value;
		return *this;
	}
	BatchNormalizationOp &eps(float value) {
		this->_eps = value;
		return *this;
	}
	BatchNormalizationOp &no_cache() {
		this->_cache = false;
		return *this;
	}
};

class LrnOp : public NodeOp<LrnOp> {
public:
	int _n = 5;
	float _alpha = 1e-4f;
	float _beta = 0.75f;
	float _k = 2.0f;
public:
	LrnOp(std::string name = "lrn") {
		this->name(name);
	}
	LrnOp &n(int size) {
		this->_n = size;
		return *this;
	}
	LrnOp &alpha(float value) {
		this->_alpha = value;
		return *this;
	}
	LrnOp &beta(float value) {
		this->_beta = value;
		return *this;
	}
	LrnOp &k(float value) {
		this->_k = value;
		return *this;
	}
};

class SioOutputOp : public NodeOp<SioOutputOp> {
public:
	std::string _host = "127.0.0.1";
	int _port = 6643;
public:
	SioOutputOp(std::string name = "sio_output") {
		this->name(name);
	}
	SioOutputOp &host(std::string address) {
		this->_host = address;
		return *this;
	}
	SioOutputOp &port(int number) {
		this->_port = number;
		return *this;
	}
};

class EqualOp : public NodeOp<EqualOp> {
public:
	EqualOp(std::string name = "equal") {
		this->name(name);
	}
};

class CastFloatOp : public NodeOp<CastFloatOp> {
public:
	CastFloatOp(std::string name = "cast_float") {
		this->name(name);
	}
};

class AccumulatorOp : public NodeOp<AccumulatorOp> {
public:
	AccumulatorOp(std::string name = "accumulator") {
		this->name(name);
	}
};

class ReplayMemoryOp : public NodeOp<ReplayMemoryOp> {
public:
	ReplayMemoryOp(std::string name = "replay_memory") {
		this->name(name);
	}
};

class BatchStddevOp : public NodeOp<BatchStddevOp> {
public:
	BatchStddevOp(std::string name = "batch_stddev") {
		this->name(name);
	}
};

class PassThroughOp : public NodeOp<PassThroughOp> {
public:
	bool _stop_gradients = false;
public:
	PassThroughOp(std::string name = "pass_through") {
		this->name(name);
	}
	PassThroughOp &stop_gradients() {
		this->_stop_gradients = true;
		return *this;
	}
};

class GaussianOp : public NodeOp<GaussianOp> {
public:
	GaussianOp(std::string name = "gaussian") {
		this->name(name);
	}
};

class GaussianKernelOp : public NodeOp<GaussianKernelOp> {
public:
	GaussianKernelOp(std::string name = "gaussian_kernel") {
		this->name(name);
	}
};

class GaussianBlurOp : public NodeOp<GaussianBlurOp> {
public:
	GaussianBlurOp(std::string name = "gaussian_blur") {
		this->name(name);
	}
};

template<class T>
class DeepFlowDllExport SolverOp {
public:
	std::string _name = "Solver";
	std::string _scope = "Default";
	float _learning_rate = 0.0002f;
public:
	T &name(std::string name) {
		this->_name = name;
		return *((T*)this);
	}
	T &scope(std::string scope) {
		this->_scope = scope;
		return *((T*)this);
	}
	T &lr(float value) {
		this->_learning_rate = value;
		return *((T*)this);
	}
};

class SgdSolverOp : public SolverOp<SgdSolverOp> {
public:
	float _momentum = 0.9f;
public:
	SgdSolverOp(std::string name = "sgd") {
		this->name(name);
		lr(0.1f);
	}
	SgdSolverOp &momentum(float value) {
		this->_momentum = value;
		return *this;
	}
};

class AdamSolverOp : public SolverOp<AdamSolverOp> {	
public:
	float _eps = 10e-8f;
	float _beta1 = 0.9f;
	float _beta2 = 0.999f;
public:
	AdamSolverOp(std::string name = "adam") {
		this->name(name);
		this->lr(0.0002f);
	}
	AdamSolverOp &beta1(float value) {
		this->_beta1 = value;
		return *this;
	}
	AdamSolverOp &beta2(float value) {
		this->_beta2 = value;
		return *this;
	}
	AdamSolverOp &eps(float value) {
		this->_eps = value;
		return *this;
	}
};

class AdaDeltaSolverOp : public SolverOp<AdaDeltaSolverOp> {
public:
	float _momentum = 0.9f;
	float _delta = 1e-6f;
public:
	AdaDeltaSolverOp(std::string name = "adelta") {
		this->name(name);
		this->lr(1.0f);
	}
	AdaDeltaSolverOp &momentum(float value) {
		this->_momentum = value;
		return *this;
	}
	AdaDeltaSolverOp &delta(float value) {
		this->_delta = value;
		return *this;
	}
};

class RmsPropSolverOp : public SolverOp<RmsPropSolverOp> {
public:
	float _rms_decay = 0.9f;
	float _eps = 0.001f;	
public:
	RmsPropSolverOp(std::string name = "rmsprop") {
		this->name(name);
		this->lr(0.0002f);
	}
	RmsPropSolverOp &decay(float value) {
		this->_rms_decay = value;
		return *this;
	}
	RmsPropSolverOp &eps(float value) {
		this->_eps = value;
		return *this;
	}
};
