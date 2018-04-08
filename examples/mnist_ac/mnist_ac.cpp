
#include "core/deep_flow.h"
#include "core/session.h"

#include <random>
#include <gflags/gflags.h>

DEFINE_string(mnist, "C:/Projects/deepflow/data/mnist", "Path to mnist folder dataset");
DEFINE_int32(batch, 100, "Batch size");
DEFINE_int32(debug, 0, "Level of debug");
DEFINE_int32(iter, 10000, "Maximum iterations");
DEFINE_string(load, "", "Load from mnist_acXXXX.bin");
DEFINE_int32(save, 1000 , "Save model iter frequency (Don't Save = 0)");
DEFINE_bool(train, false, "Train");
DEFINE_bool(test, false, "Test");
DEFINE_int32(bottleneck, 16, "Bottleneck Channel Size");

void load_session(DeepFlow *df, std::string prefix, std::string suffix) {
	std::string filename = prefix + suffix + ".bin";
	std::cout << "Loading "<< prefix << " from " << filename << std::endl;	
	df->block()->load_from_binary(filename);	
}

std::string deconv(DeepFlow *df, std::string input, std::string solver, int input_channels, int output_channels, int kernel, int pad, int stride, bool activation, std::string name) {

	auto node = input;	
	node = df->transposed_conv2d(node, input_channels, output_channels,kernel, pad, stride, false, solver, name + "_conv");	
	if (activation) {
		node = df->batch_normalization(node, solver, output_channels, 0.001f, name + "_bn");
		return df->relu(node);	
	}
	return node;
}

std::string conv(DeepFlow *df, std::string input, std::string solver, int input_channels, int output_channels, int kernel, int pad, int stride, bool activation, std::string name) {	
	auto node = df->conv2d(input, input_channels, output_channels, kernel, pad, stride, true, solver, name + "_conv");	
	if (activation) {		
		return df->leaky_relu(node, 0.2, name + "_relu");
	}
	return node;
}

void create_graph(DeepFlow *df) {	

	df->mnist_reader(FLAGS_mnist, FLAGS_batch, MNISTReader::MNISTReaderType::Train, MNISTReader::Data, "mnist_train_data");
	df->mnist_reader(FLAGS_mnist, FLAGS_batch, MNISTReader::MNISTReaderType::Test, MNISTReader::Data, "mnist_test_data");
	auto disp_input = df->place_holder({ FLAGS_batch, 1, 28, 28 }, Tensor::Float, "disp_input");
	df->display(disp_input, 1, DeepFlow::VALUES, 1, true, "disp");
	auto loss_input = df->place_holder({ FLAGS_batch, 1, 28, 28 }, Tensor::Float, "loss_input");
	auto loss_target = df->place_holder({ FLAGS_batch, 1, 28, 28 }, Tensor::Float, "loss_target");
	auto sqerr = df->square_error(loss_input, loss_target);
	auto loss = df->loss(sqerr, DeepFlow::AVG, 1.0f, 0.0f, "loss");

	int fn = 64;

	auto solver = df->adam_solver(0.0002f, 0.5f, 0.99f);

	auto enc_input = df->place_holder({ FLAGS_batch, 1, 28, 28 }, Tensor::Float, "enc_input");	

	auto node = conv(df, enc_input, solver, 1, fn, 3, 1, 2, true, "e1");
	node = conv(df, node, solver, fn, fn * 2, 3, 1, 2, true, "e2");
	node = conv(df, node, solver, fn * 2, fn * 4, 3, 1, 2, true, "e2");
	node = df->dense(node, { (fn * 4) * 4 * 4, FLAGS_bottleneck, 1, 1 }, true, solver, "e3");
	node = df->tanh(node, "h");
	node = df->dense(node, { FLAGS_bottleneck, fn * 4, 4, 4 }, true, solver, "d1");	
	node = deconv(df, node, solver, fn * 4, fn * 2, 2, 1, 2, 1, "d3");
	node = deconv(df, node, solver, fn * 2, fn    , 3, 1, 2, 1, "d4");
	node = deconv(df, node, solver, fn    , 1      , 3, 1, 2, 0, "d5");

	df->pass_through(node, false, "dec_output");
	
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
		load_session(&df,"mnist_ac", FLAGS_load);

	std::shared_ptr<Session> session = df.session();
	session->initialize(execution_context);
	
	auto mnist_train_data = session->get_node("mnist_train_data");
	auto mnist_test_data = session->get_node("mnist_test_data");
	auto enc_input = session->get_placeholder("enc_input");
	auto dec_output = session->get_node("dec_output");
	auto disp_input = session->get_placeholder("disp_input");
	auto disp = session->get_node("disp");
	auto loss_input = session->get_placeholder("loss_input");
	auto loss_target = session->get_placeholder("loss_target");
	auto loss = session->get_node("loss");
	
	if (FLAGS_train) {		
		int iter;
		for (iter = 1; iter <= FLAGS_iter && execution_context->quit != true; ++iter) {

			execution_context->execution_mode = ExecutionContext::TRAIN;
			execution_context->current_epoch = iter;
			execution_context->current_iteration = iter;

			session->forward({ mnist_train_data });
			session->forward({ dec_output }, { {enc_input, mnist_train_data->output(0)->value() } });
			session->forward({ loss }, {
				{ loss_input , dec_output->output(0)->value() },
				{ loss_target , mnist_train_data->output(0)->value() }
			});
			session->backward({ loss });
			session->backward({ dec_output }, { { dec_output , loss_input->output(0)->diff() } });
			session->apply_solvers();			

			execution_context->execution_mode = ExecutionContext::TEST;
			session->forward({ mnist_test_data });
			session->forward({ dec_output }, { { enc_input, mnist_test_data->output(0)->value() } });
			session->forward({ disp }, { { disp_input , dec_output->output(0)->value() } });

			if (FLAGS_save != 0 && iter % FLAGS_save == 0) {
				session->save("mnist_ac" + std::to_string(iter) + ".bin");
			}

		}

		if (FLAGS_save != 0) {
			session->save("mnist_ac" + std::to_string(iter) + ".bin");
		}
	}

	if (FLAGS_test) {
		execution_context->execution_mode = ExecutionContext::TEST;
		int iter;
		for (iter = 1; iter <= FLAGS_iter && execution_context->quit != true; ++iter) {
			session->forward({ mnist_test_data });
			session->forward({ dec_output }, { { enc_input, mnist_test_data->output(0)->value() } });
			session->forward({ disp }, { { disp_input , dec_output->output(0)->value() } });
		}
	}

	cudaDeviceReset();

}
