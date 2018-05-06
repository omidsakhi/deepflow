#pragma once

#include "core/export.h"

#include "core/tensor.h"
#include "core/optional_params.h"

#include "core/caffe.h"
#include "nodes/block.h"

#include <string>
#include <memory>
#include <list>

class Session;

class DeepFlowDllExport DeepFlow : public std::enable_shared_from_this<DeepFlow> {
	friend class Session;
	friend class Caffe;
public:

	DeepFlow();
	DeepFlow(std::shared_ptr<Block> block);

	// GENERATORS
	std::string mnist_reader(std::string folder_path, MNISTReaderOp &params = MNISTReaderOp());
	std::string data_generator(std::string initializer, DataGeneratorOp &params = DataGeneratorOp());
	std::array<std::string, 2> text_image_generator(std::string initializer, TextImageGeneratorOp &params = TextImageGeneratorOp());
	std::string imread(std::string file_path, ImreadOp &params = ImreadOp());
	std::string imbatch(std::string folder_path, std::initializer_list<int> dims, ImbatchOp &params = ImbatchOp());

	// INITIALIZERS
	std::string fill(std::initializer_list<int> dims, float value, std::string name = "fill");
	std::string index_fill(std::initializer_list<int> dims, float offset, std::string name = "index_fill");
	std::string zeros(std::initializer_list<int> dims, std::string name = "zeros");
	std::string ones(std::initializer_list<int> dims, std::string = "ones");
	std::string random_uniform(std::initializer_list<int> dims, float min, float max, std::string name = "random_uniform");
	std::string random_normal(std::initializer_list<int> dims, float mean, float stddev, std::string name = "random_normal");
	std::string truncated_normal(std::initializer_list<int> dims, float mean, float stddev, std::string name = "truncated_normal");
	std::string step(std::initializer_list<int> dims, float min, float max, std::string name = "step");
	std::string three_state(std::initializer_list<int> dims, std::string name = "three_state");
	std::string gradient(std::initializer_list<int> dims, std::string name = "gradient");
	std::string constant(std::initializer_list<int> dims, std::vector<float> values, std::string name = "constant");

	// VARIABLES & PLACE HOLDER
	std::string variable(std::string initializer, std::string solver = "", VariableOp &params = VariableOp());
	std::string place_holder(std::array<int,4> dims, PlaceholderOp &params = PlaceholderOp());

	//CONVOLUTION
	std::string conv2d(std::string input, std::string filter, ConvolutionOp &params = ConvolutionOp());
	std::string conv2d(std::string input, int input_channels, int output_channels, std::string solver, ConvolutionOp &params = ConvolutionOp());
	std::string transposed_conv2d(std::string input, std::string filter, ConvolutionOp &params = ConvolutionOp());
	std::string transposed_conv2d(std::string input, int input_channels, int output_channels, std::string solver, ConvolutionOp &params = ConvolutionOp());

	// RESTRUCTURE
	std::string restructure(std::string input, int first_dims, int second_dim, RestructureOp &params = RestructureOp());
	std::string pooling(std::string input, PoolingOp &params = PoolingOp());
	std::string lifting(std::string input, LiftingOp &params = LiftingOp());	
	std::string patching(std::string input, PatchingOp &params = PatchingOp());
	std::string patch_sampling(std::string input, int patch_width, int patch_height, PatchSamplingOp &params = PatchSamplingOp());
	std::string resize(std::string input, float height_scale, float width_scale, ResizeOp &params = ResizeOp());
	std::string concate(std::list<std::string> inputs, ConcateOp &params = ConcateOp());
	std::string reshape(std::string input, std::array<int, 4> output_dims, ReshapeOp &params = ReshapeOp());	
	std::string spatial_transformer(std::string input, std::string theta, std::string grid, SpatialTransformerOp &params = SpatialTransformerOp());
	std::string spatial_transformer(std::string input, std::string theta, int n, int h, int w, std::string solver, SpatialTransformerOp &params = SpatialTransformerOp());

	// MATH
	std::string add(std::string a, std::string b, AddOp &params = AddOp());
	std::string subtract(std::string a, std::string b, SubtractOp &params = SubtractOp());
	std::string dot(std::string a, std::string b, DotOp &params = DotOp());
	std::string square(std::string a, SquareOp &params = SquareOp());
	std::string abs(std::string a, AbsOp &params = AbsOp());
	std::string exp(std::string a, ExpOp &params = ExpOp());
	std::string log(std::string a, LogOp &params = LogOp());		
	std::string square_error(std::string a, std::string b, SquareErrorOp &params = SquareErrorOp());
	std::string softmax(std::string a, SoftmaxOp &params = SoftmaxOp());
	std::string max(std::string a, std::string b, MaxOp &params = MaxOp());
	std::string nand(std::string a, std::string b, NandOp &params = NandOp());

	// matmul
	std::string matmul(std::string a, std::string b, MatmulOp &params = MatmulOp());
	std::string dense(std::string input, std::initializer_list<int> dims, std::string solver, DenseOp &params = DenseOp());

	// NEURAL NETS
	std::string bias_add(std::string input, std::string weights, BiasAddOp &params = BiasAddOp());
	std::string bias_add(std::string input, int output_channels, std::string solver, BiasAddOp &params = BiasAddOp());
	std::string dropout(std::string a, DropoutOp &params = DropoutOp());

	//REDUCTION
	
	std::string argmax(std::string input, int reduce_dimension, ArgmaxOp &params = ArgmaxOp());
	std::string argmin(std::string input, int reduce_dimension, ArgminOp &params = ArgminOp());
	std::string reduce_max(std::string input, int reduce_dimension, ReduceMaxOp &params = ReduceMaxOp());
	std::string reduce_min(std::string input, int reduce_dimension, ReduceMinOp &params = ReduceMinOp());
	std::string reduce_mean(std::string input, int reduce_dimension, ReduceMeanOp &params = ReduceMeanOp());
	std::string reduce_sum(std::string input, int reduce_dimension, ReduceSumOp &params = ReduceSumOp());
	std::string reduce_amax(std::string input, int reduce_dimension, ReduceAmaxOp &params = ReduceAmaxOp());
	std::string reduce_norm1(std::string input, int reduce_dimension, ReduceNorm1Op &params = ReduceNorm1Op());
	std::string reduce_norm2(std::string input, int reduce_dimension, ReduceNorm2Op &params = ReduceNorm2Op());
	std::string reduce_all(std::string input, ReduceAllOp &params = ReduceAllOp());	

	//ACTIVATIONS
	std::string leaky_relu(std::string a, LeakyReluOp &params = LeakyReluOp());
	std::string sigmoid(std::string a, SigmoidOp &params = SigmoidOp());
	std::string relu(std::string a, ReluOp &params = ReluOp());
	std::string tanh(std::string a, TanhOp &params = TanhOp());
	std::string clipped_relu(std::string a, ClippedReluOp &params = ClippedReluOp());
	std::string elu(std::string a, EluOp &params = EluOp());
	std::string prelu(std::string input, std::string w, PReluOp &params = PReluOp());
	std::string prelu(std::string input, int output_channels, std::string solver, PReluOp &params = PReluOp());
	std::string dprelu(std::string input, std::string enabler, DPReluOp &params = DPReluOp());

	// SELECTORS	
	std::string random_selector(std::string input_1, std::string input_2, RandomSelectorOp &params = RandomSelectorOp());
	std::string multiplexer(std::list<std::string> inputs, MultiplexerOp &params = MultiplexerOp());
	std::string switcher(std::string input, SwitchOp &params = SwitchOp());
				
	// SOLVERS
	std::string sgd_solver( SgdSolverOp &params = SgdSolverOp());	
	std::string adam_solver( AdamSolverOp &params = AdamSolverOp());
	std::string adadelta_solver( AdaDeltaSolverOp &params = AdaDeltaSolverOp());
	std::string rmsprop_solver( RmsPropSolverOp &params = RmsPropSolverOp());
	
	// PRINTERS & DISPLAYS
	void print(std::list<std::string> inputs, std::string message, PrintOp &params = PrintOp());
	void logger(std::list<std::string> inputs, std::string file_path, std::string message, LoggerOp &params = LoggerOp());
	void display(std::string input, DisplayOp &params = DisplayOp());
	void imwrite(std::string input, std::string filename, ImwriteOp &params = ImwriteOp());
	void psnr(std::string a, std::string b, PsnrOp &params = PsnrOp());	

	// NORMALIZATION
	std::string batch_normalization(std::string input, std::string scale, std::string bias, BatchNormalizationOp &params = BatchNormalizationOp());
	std::string batch_normalization(std::string input, int output_channels, std::string solver, BatchNormalizationOp &params = BatchNormalizationOp());
	std::string batch_normalization(std::string input, int output_channels, int output_height, int output_width, std::string solver, BatchNormalizationOp &params = BatchNormalizationOp());
	std::string lrn(std::string input, LrnOp &params = LrnOp());

	// SOCKET IO
	void sio_output(std::list<std::string> inputs, SioOutputOp &params = SioOutputOp());

	// OTHER	
	std::string loss(std::string input, LossOp &params = LossOp());
	std::string equal(std::string a, std::string b, EqualOp &params = EqualOp());	
	std::array<std::string,2> accumulator(std::string input, AccumulatorOp &params = AccumulatorOp());
	std::string replay_memory(std::string input, int capacity, ReplayMemoryOp &params = ReplayMemoryOp());
	std::string batch_stddev(std::string input, BatchStddevOp &params = BatchStddevOp());
	std::string pass_through(std::string input, PassThroughOp &params = PassThroughOp());
	std::string gaussian(std::string mean, std::string sigma, GaussianOp &params = GaussianOp());
	std::string gaussian_kernel(int window_size, float sigma, GaussianKernelOp &params = GaussianKernelOp());
	std::string gabor_kernel(GaborKernelOp & params = GaborKernelOp());
	std::string gaussian_blur(std::string input, int window_size, float sigma, GaussianBlurOp &params = GaussianBlurOp());	
	std::string identify(std::string input, int input_channels, int output_channels, float scale, std::string scope);

	std::shared_ptr<Block> block();
	std::shared_ptr<Session> session();

	// CAFFE
	void load_from_caffe_model(std::string file_path, std::initializer_list<std::pair<std::string, std::array<int, 4>>> inputs, bool verbose = false);

	void with(std::string scope);
private:
	std::string _reduce(std::string input, ReduceOp &params = ReduceOp());		

private:		
	std::shared_ptr<Block> _block;
	std::string _scope = "Default";
};