
#include "core/deep_flow.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_string(mnist, "./data/mnist", "Path to MNIST data folder");
DEFINE_string(i, "", "Trained network model to load");
DEFINE_string(o, "", "Trained network model to save");
DEFINE_bool(text, false, "Save model as text");
DEFINE_bool(includeweights, false, "Also save weights in text mode");
DEFINE_bool(includeinits, false, "Also save initial values");
DEFINE_int32(batch, 100, "Batch size");
DEFINE_string(run,"", "Phase to execute graph");
DEFINE_bool(printiter, true, "Print iteration message");
DEFINE_bool(printepoch, false, "Print epoch message");

void main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	//CudaHelper::setOptimalThreadsPerBlock();	

	int batch_size = FLAGS_batch;
	
	DeepFlow df;
		
	if (FLAGS_i.empty()) {
		df.define_phase("Train", PhaseParam_PhaseBehaviour_TRAIN);
		df.define_phase("Test", PhaseParam_PhaseBehaviour_TEST);
		auto mnist_trainset = df.mnist_reader(FLAGS_mnist, batch_size, MNISTReaderType::Train, "trainset", { "Train" });
		auto mnist_testset = df.mnist_reader(FLAGS_mnist, batch_size, MNISTReaderType::Test, "testset", { "Test" });
		auto px = df.phaseplexer(mnist_trainset->output(0), "Train", mnist_testset->output(0), "Test", "px");		
		auto py = df.phaseplexer(mnist_trainset->output(1), "Train", mnist_testset->output(1), "Test", "py");
		auto f1 = df.variable(df.random_uniform({ 20, 1 , 5, 5 }, -0.1f, 0.1f), "f1");
		auto conv1 = df.conv2d(px, f1, 2, 2, 1, 1, 1, 1, "conv1");
		auto pool1 = df.pooling(conv1, 2, 2, 0, 0, 2, 2, "pool1");
		auto f2 = df.variable(df.random_uniform({ 50, 20 , 5, 5 }, -0.1f, 0.1f), "f2");
		auto conv2 = df.conv2d(pool1, f2, "conv2");
		auto pool2 = df.pooling(conv2, 2, 2, 0, 0, 2, 2, "pool2");
		auto w1 = df.variable(df.random_uniform({ 1250, 500, 1 , 1 }, -0.1f, 0.1f), "w1");
		auto b1 = df.variable(df.step({ 1, 500, 1 , 1 }, -1.0f, 1.0f), "b1");
		auto m1 = df.matmul(pool2, w1,"m1");
		auto bias1 = df.bias_add(m1, b1,"bias1");
		auto relu1 = df.relu(bias1, -0.01f, "relu1");
		auto dropout = df.dropout(relu1, 0.5f, "dropout");
		auto w2 = df.variable(df.random_uniform({ 500, 10, 1 , 1 }, -0.1f, 0.1f), "w2");
		auto b2 = df.variable(df.step({ 1, 10, 1, 1 }, -1.0f, 1.0f), "b2");
		auto m2 = df.matmul(dropout, w2,"m2");
		auto bias2 = df.bias_add(m2, b2,"bias2");
		auto relu2 = df.relu(bias2, -0.01f, "relu2");
		auto loss = df.softmax_loss(relu2, py, "loss");
		auto mse = df.reduce_mean(df.reduce_mean(df.square(loss), 1), 0, "mse", {"Train"});
		auto correct_class = df.argmax(py, 1, "correct_class", {"Train"});
		auto predict_class = df.argmax(loss->node()->output(1), 1, "predict_class");
		auto accuracy = df.reduce_mean(df.equal(predict_class, correct_class), 0, "accuracy", {"Train"});
		auto print = df.print({ mse, accuracy }, "    MSE: {0} - ACCURACY: {1}\n","print");
		df.set_solver(df.gain_solver(loss, 2000, 0.9999f, 0.0001f, 100, 0.1f, 0.05f, 0.95f));
		df.global_node_initializer();
	}			

	if (!FLAGS_run.empty()) {
		df.run(FLAGS_run, FLAGS_printiter, FLAGS_printepoch);
	}

	if (!FLAGS_o.empty())
	{
		if (FLAGS_text)
			df.save_as_text(FLAGS_o, FLAGS_includeweights, FLAGS_includeinits);
		else
			df.save_as_binary(FLAGS_o, FLAGS_includeinits);
	}

}

