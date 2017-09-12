
#include "core/deep_flow.h"
#include "core/session.h"

#include <gflags/gflags.h>

DEFINE_string(mnist, "C:/Projects/deepflow/data/mnist", "Path to MNIST data folder");
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

		df.define_phase("Train", DeepFlow::TRAIN);
		df.define_phase("Validation", DeepFlow::VALIDATION);

		auto solver = df.adam_solver(0.0002f, 0.5f);
		//auto solver = df.gain_solver(0.999900, 0.000100, 10.000000, 0.000000, 0.050000, 0.950000, "solver");
		//auto solver = df.adadelta_solver();

		auto test_data = df.mnist_reader(FLAGS_mnist, 100, MNISTReader::Test, MNISTReader::Data, "test_data", { "Validation" });		
		auto train_data = df.mnist_reader(FLAGS_mnist, 100, MNISTReader::Train, MNISTReader::Data, "train_data", { "Train" });
		auto data_selector = df.phaseplexer(train_data, "Train", test_data, "Validation", "data_selector", {});
				
		
		auto conv1_w = df.variable(df.random_uniform({ 16, 1, 5, 5 }, -0.100000, 0.100000), solver, "conv1_w", {});
		auto conv1 = df.conv2d(data_selector, conv1_w, 2, 2, 2, 2, 1, 1, "conv1", {});		
		auto conv1_r = df.elu(conv1, 0.01);		

		auto conv2_w = df.variable(df.random_uniform({ 32, 16, 5, 5 }, -0.100000, 0.100000), solver, "conv2_w", {});
		auto conv2 = df.conv2d(conv1_r, conv2_w, 2, 2, 2, 2, 1, 1, "conv2", {});
		auto conv2_r = df.elu(conv2, 0.01);

		auto conv3_w = df.variable(df.random_uniform({ 64, 32, 5, 5 }, -0.100000, 0.100000), solver, "conv3_w", {});
		auto conv3 = df.conv2d(conv2_r, conv3_w, 2, 2, 2, 2, 1, 1, "conv3", {});
		auto conv3_r = df.elu(conv3, 0.01);

		auto w1 = df.variable(df.random_uniform({ 1024, 500, 1, 1 }, -0.100000, 0.100000), solver, "w1", {});
		auto m1 = df.matmul(conv3_r, w1, "m1", {});
		

		/*
		auto conv1_w = df.variable(df.random_uniform({ 20, 1, 5, 5 }, -0.100000, 0.100000), solver, "conv1_w", {});		
		auto conv1 = df.conv2d(data_selector, conv1_w, 2, 2, 1, 1, 1, 1, "conv1", {});
		auto pool1 = df.pooling(conv1, 2, 2, 0, 0, 2, 2, "pool1", {});
		auto conv2_w = df.variable(df.random_uniform({ 50, 20, 5, 5 }, -0.100000, 0.100000), solver, "conv2_w", {});		
		auto conv2 = df.conv2d(pool1, conv2_w, 0, 0, 1, 1, 1, 1, "conv2", {});
		auto pool2 = df.pooling(conv2, 2, 2, 0, 0, 2, 2, "pool2", {});
		auto w1 = df.variable(df.random_uniform({ 1250, 500, 1, 1 }, -0.100000, 0.100000), solver, "w1", {});
		auto m1 = df.matmul(pool2, w1, "m1", {});
		*/


		
		auto b1 = df.variable(df.step({ 1, 500, 1, 1 }, 0, 1.000000), solver, "b1", {});
		auto bias1 = df.bias_add(m1, b1, "bias1", {});
		auto relu1 = df.leaky_relu(bias1, 0.010000, "relu1", {});
		auto dropout = df.dropout(relu1, 0.500000, true, "dropout", {});
		auto w2 = df.variable(df.random_uniform({ 500, 10, 1, 1 }, -0.100000, 0.100000), solver, "w2");
		auto m2 = df.matmul(dropout, w2, "m2");
		auto b2 = df.variable(df.step({ 1, 10, 1, 1 }, 0, 1.000000), solver, "b2");
		auto bias2 = df.bias_add(m2, b2, "bias2");
		auto relu2 = df.leaky_relu(bias2, 0.010000, "relu2");
		auto softmax = df.softmax(relu2);
		auto softmax_log = df.log(softmax);
		auto train_labels = df.mnist_reader(FLAGS_mnist, 100, MNISTReader::Train, MNISTReader::Labels, "train_labels", { "Train" });
		auto test_labels = df.mnist_reader(FLAGS_mnist, 100, MNISTReader::Test, MNISTReader::Labels, "test_labels", { "Validation" });
		auto label_selector = df.phaseplexer(train_labels, "Train", test_labels, "Validation", "label_selector", {});		
		auto softmax_dot = df.dot(softmax_log, label_selector);
		auto loss = df.loss(softmax_dot, DeepFlow::AVG, "loss");
		auto predict = df.argmax(softmax, 1, "predict", {});
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

