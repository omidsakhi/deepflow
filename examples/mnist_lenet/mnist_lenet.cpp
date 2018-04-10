
#include "core/deep_flow.h"
#include "core/session.h"

#include <gflags/gflags.h>

DEFINE_string(mnist, "C:/Projects/deepflow/data/mnist", "Path to mnist folder dataset");
DEFINE_int32(batch, 100, "Batch size");
DEFINE_int32(debug, 0, "Level of debug");
DEFINE_int32(epoch, 10000, "Maximum epoch");
DEFINE_string(load, "", "Load from mnist_acXXXX.bin");
DEFINE_int32(save, 10, "Save model epoch frequency (Don't Save = 0)");
DEFINE_bool(train, false, "Train");

void load_session(DeepFlow *df, std::string filename) {
	filename += ".bin";
	std::cout << "Loading from " << filename << std::endl;
	df->block()->load_from_binary(filename);
}

void create_graph(DeepFlow *df) {

	auto solver = df->adam_solver(0.0001f, 0.5f, 0.99f);
	//auto solver = df->sgd_solver(0.01f, 0.01f);

	df->mnist_reader(FLAGS_mnist, FLAGS_batch, MNISTReader::MNISTReaderType::Train, MNISTReader::Data, "mnist_train_data");
	df->mnist_reader(FLAGS_mnist, FLAGS_batch, MNISTReader::MNISTReaderType::Test, MNISTReader::Data, "mnist_test_data");
	df->mnist_reader(FLAGS_mnist, FLAGS_batch, MNISTReader::Train, MNISTReader::Labels, "mnist_train_labels");
	df->mnist_reader(FLAGS_mnist, FLAGS_batch, MNISTReader::Test, MNISTReader::Labels, "mnist_test_labels");

	auto loss_softmax_input = df->place_holder({ FLAGS_batch, 10, 1, 1 }, Tensor::Float, "loss_softmax_input");
	auto loss_target_labels = df->place_holder({ FLAGS_batch, 10, 1, 1 }, Tensor::Float, "loss_target_labels");
	auto loss_softmax_log = df->log(loss_softmax_input, -1.0f, "loss_softmax_log");
	auto loss_softmax_dot = df->dot(loss_softmax_log, loss_target_labels);
	auto loss = df->loss(loss_softmax_dot, DeepFlow::AVG, 1.0f, 0.0f, "loss");

	auto pred_softmax_input = df->place_holder({ FLAGS_batch, 10, 1, 1 }, Tensor::Float, "pred_softmax_input");
	auto pred_target_labels = df->place_holder({ FLAGS_batch, 10, 1, 1 }, Tensor::Float, "pred_target_labels");
	auto max_actual = df->argmax(pred_softmax_input, 1, "pred_actual");
	auto max_target = df->argmax(pred_target_labels, 1, "pred_target");	
	auto equal = df->equal(max_actual, max_target, "equal");
	auto acc = df->accumulator(equal, "accumulator");
	auto correct_count = df->reduce_sum(acc[0], 0, "correct_count");

	int fn = 64; 

	auto net = df->place_holder({ FLAGS_batch, 1, 28, 28 }, Tensor::Float, "input");
	net = df->conv2d(net, 1, fn, 3, 1, 2, true, solver, "conv1"); // 14x14
	net = df->leaky_relu(net, 0.2f);	
	net = df->conv2d(net, fn, fn * 2, 3, 1, 2, true, solver, "conv2"); // 7x7
	net = df->leaky_relu(net, 0.2f);
	net = df->conv2d(net, fn * 2, fn * 4, 3, 1, 2, true, solver, "conv3"); // 4x4
	net = df->leaky_relu(net, 0.2f);
	net = df->dense(net, { (fn * 4) * 4 * 4, 512, 1, 1 }, true, solver, "fc1");
	//net = df->dropout(net);
	net = df->dense(net, { 512, 10, 1, 1 }, true, solver, "fc2");
	net = df->softmax(net, DeepFlow::SoftmaxMode::INSTANCE, "output");

}

void main(int argc, char** argv) {

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();

	auto execution_context = std::make_shared<ExecutionContext>();
	execution_context->debug_level = FLAGS_debug;

	DeepFlow df;
	if (FLAGS_load.empty())
		create_graph(&df);
	else
		load_session(&df, "mnist_lenet" + FLAGS_load);

	std::shared_ptr<Session> session = df.session();
	session->initialize(execution_context);
		
	auto mnist_train_data = session->get_node("mnist_train_data");
	auto mnist_test_data = session->get_node("mnist_test_data");
	auto mnist_train_labels = session->get_node("mnist_train_labels");
	auto mnist_test_labels = session->get_node("mnist_test_labels");
	auto loss_softmax_input = session->get_placeholder("loss_softmax_input");
	auto loss_target_labels = session->get_placeholder("loss_target_labels");
	auto loss = session->get_node("loss");	
	auto input = session->get_placeholder("input");
	auto output = session->get_node("output");
	auto correct_count = session->get_node("correct_count");
	auto pred_softmax_input = session->get_placeholder("pred_softmax_input");
	auto pred_target_labels = session->get_placeholder("pred_target_labels");
	auto acc = session->get_node<Accumulator>("accumulator");

	if (FLAGS_train) {		
		int epoch = 0;		
		
		for (; epoch <= FLAGS_epoch && execution_context->quit != true; ++epoch) {
			execution_context->current_epoch = epoch;			
			std::cout << "Epoch: [" << epoch << "/" << FLAGS_epoch << "]\n";
			acc->reset();
			do {				execution_context->execution_mode = ExecutionContext::TEST;
				session->forward({ mnist_test_data, mnist_test_labels });
				session->forward({ output }, { { input , mnist_test_data->output(0)->value() } });
				session->forward({ correct_count }, { { pred_softmax_input, output->output(0)->value() },{ pred_target_labels, mnist_test_labels->output(0)->value() } });
			} while (!mnist_test_data->is_last_batch());
			{
				auto num_correct = correct_count->output(0)->value()->toFloat();
				auto num_total = acc->output(1)->value()->toFloat();
				std::cout << " Test accuracy: " << num_correct << " out of " << num_total << " - " << num_correct / num_total * 100.0f << "%" << std::endl;
			}
			acc->reset();
			do {			
				execution_context->execution_mode = ExecutionContext::TRAIN;
				execution_context->current_iteration = epoch;
				session->forward({ mnist_train_data, mnist_train_labels });					
				session->forward({ output }, { { input , mnist_train_data->output(0)->value() } });
				session->forward({ loss }, { { loss_softmax_input, output->output(0)->value() } , { loss_target_labels, mnist_train_labels->output(0)->value() } });
				session->forward({ correct_count }, { { pred_softmax_input, output->output(0)->value() },{ pred_target_labels, mnist_train_labels->output(0)->value() } });
				session->backward({ loss });
				session->backward({ output }, { { output, loss_softmax_input->output(0)->diff() } });
				session->apply_solvers();
			} while (!mnist_train_data->is_last_batch());
			{
				auto num_correct = correct_count->output(0)->value()->toFloat();
				auto num_total = acc->output(1)->value()->toFloat();
				std::cout << "Train accuracy: " << num_correct << " out of " << num_total << " - " << num_correct / num_total * 100.0f << "%" << std::endl;
			}
			if (FLAGS_save != 0 && epoch % FLAGS_save == 0) {
				session->save("mnist_lenet" + std::to_string(epoch) + ".bin");
			}
		}
	}
	
}

