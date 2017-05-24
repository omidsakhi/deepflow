
#include "core/deep_flow.h"
#include "core/session.h"

#include <gflags/gflags.h>

DEFINE_string(mnist, "./data/mnist", "Path to MNIST data folder");
DEFINE_string(i, "", "Trained network model to load");
DEFINE_string(o, "", "Trained network model to save");
DEFINE_bool(text, false, "Save model as text");
DEFINE_bool(includeweights, false, "Also save weights in text mode");
DEFINE_bool(includeinits, false, "Also save initial values");
DEFINE_int32(batch, 100, "Batch size");
DEFINE_string(run,"", "Phase to execute graph");
DEFINE_bool(printiter, false, "Print iteration message");
DEFINE_bool(printepoch, true, "Print epoch message");
DEFINE_int32(debug, 0, "Level of debug");
DEFINE_int32(epoch, 1000, "Maximum epochs");
DEFINE_int32(iter, -1, "Maximum iterations");
DEFINE_bool(cpp, false, "Print C++ code");


void main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();	

	int batch_size = FLAGS_batch;
	
	DeepFlow df;
		
	if (FLAGS_i.empty()) {		

		df.define_phase("Train", deepflow::PhaseParam_PhaseBehaviour_TRAIN);
		df.define_phase("Validation", deepflow::PhaseParam_PhaseBehaviour_VALIDATION);

		auto gain = df.gain_solver(0.999900, 0.000100, 10.000000, 0.000000, 0.050000, 0.950000, "gain");

		auto test_data = df.mnist_reader("./data/mnist", 100, MNISTReader::Test, MNISTReader::Data, "test_data", { "Validation" });		
		auto train_data = df.mnist_reader("./data/mnist", 100, MNISTReader::Train, MNISTReader::Data, "train_data", { "Train" });
		auto data_selector = df.phaseplexer(train_data, "Train", test_data, "Validation", "data_selector", {});
		auto conv1_w = df.variable(df.random_uniform({ 20, 1, 5, 5 }, -0.100000, 0.100000), gain, "conv1_w", {});
		auto conv1_b = df.variable(df.random_uniform({ 1, 20, 1, 1 }, -0.100000, 0.100000), gain, "conv1_b", {});
		auto conv1 = df.conv2d(data_selector, conv1_w, conv1_b, 2, 2, 1, 1, 1, 1, "conv1", {});
		auto pool1 = df.pooling(conv1, 2, 2, 0, 0, 2, 2, "pool1", {});
		auto conv2_w = df.variable(df.random_uniform({ 50, 20, 5, 5 }, -0.100000, 0.100000), gain, "conv2_w", {});
		auto conv2_b = df.variable(df.random_uniform({ 1, 50, 1, 1 }, -0.100000, 0.100000), gain, "conv2_b", {});
		auto conv2 = df.conv2d(pool1, conv2_w, conv2_b, 0, 0, 1, 1, 1, 1, "conv2", {});
		auto pool2 = df.pooling(conv2, 2, 2, 0, 0, 2, 2, "pool2", {});
		auto w1 = df.variable(df.random_uniform({ 1250, 500, 1, 1 }, -0.100000, 0.100000), gain, "w1", {});
		auto m1 = df.matmul(pool2, w1, "m1", {});
		auto b1 = df.variable(df.step({ 1, 500, 1, 1 }, -1.000000, 1.000000), gain, "b1", {});
		auto bias1 = df.bias_add(m1, b1, "bias1", {});
		auto relu1 = df.leaky_relu(bias1, 0.010000, "relu1", {});
		auto dropout = df.dropout(relu1, 0.500000, true, "dropout", {});
		auto w2 = df.variable(df.random_uniform({ 500, 10, 1, 1 }, -0.100000, 0.100000), gain, "w2", {});
		auto m2 = df.matmul(dropout, w2, "m2", {});
		auto b2 = df.variable(df.step({ 1, 10, 1, 1 }, -1.000000, 1.000000), gain, "b2", {});
		auto bias2 = df.bias_add(m2, b2, "bias2", {});
		auto relu2 = df.leaky_relu(bias2, 0.010000, "relu2", {});
		auto train_labels = df.mnist_reader("./data/mnist", 100, MNISTReader::Train, MNISTReader::Labels, "train_labels", { "Train" });
		auto test_labels = df.mnist_reader("./data/mnist", 100, MNISTReader::Test, MNISTReader::Labels, "test_labels", { "Validation" });
		auto label_selector = df.phaseplexer(train_labels, "Train", test_labels, "Validation", "label_selector", {});
		auto loss = df.softmax_loss(relu2, label_selector, "loss", {});
		auto predict = df.argmax(loss, 1, "predict", {});
		auto target = df.argmax(label_selector, 1, "target", {});
		auto equal = df.equal(predict, target, "equal", {});
		auto acc = df.accumulator(equal, DeepFlow::END_OF_EPOCH, "acc", {});
		auto correct = df.reduce_sum(acc[0], 0, "correct", {});
		df.print({ correct, acc[1] }, "    TRAINSET   CORRECT COUNT : {0} OUT OF {1} = %{}%\n", DeepFlow::END_OF_EPOCH, DeepFlow::VALUES, "print1", { "Train" });
		df.print({ correct, acc[1] }, "    VALIDATION CORRECT COUNT : {0} OUT OF {1} = %{}%\n", DeepFlow::END_OF_EPOCH, DeepFlow::VALUES, "print2", { "Validation" });
		
	}
	else {
		df.block()->load_from_binary(FLAGS_i);	
	}
	
	df.block()->print_node_params();
	df.block()->print_phase_params();

	auto session = df.session();	

	if (!FLAGS_run.empty()) {
		session->initialize();
		session->run(FLAGS_run, FLAGS_epoch, FLAGS_iter, FLAGS_printiter, FLAGS_printepoch, FLAGS_debug);
	}

	if (FLAGS_cpp) {
		session->initialize();
		std::cout << session->to_cpp() << std::endl;
	}

	
	if (!FLAGS_o.empty())
	{
		if (FLAGS_text)
			df.block()->save_as_text(FLAGS_o);
		else
			df.block()->save_as_binary(FLAGS_o);
	}
	
}

