
#include "core/deep_flow.h"
#include "core/session.h"

#include <random>
#include <gflags/gflags.h>

DEFINE_string(faces, "C:/Projects/deepflow/data/celeba128", "Path to face 128x128 dataset folder");
DEFINE_int32(batch, 40, "Batch size");
DEFINE_int32(debug, 0, "Level of debug");
DEFINE_int32(iter, 10000, "Maximum iterations");
DEFINE_string(load, "", "Load from face_acXXXX.bin");
DEFINE_int32(save, 1000 , "Save model iter frequency (Don't Save = 0)");
DEFINE_bool(train, false, "Train");
DEFINE_bool(test, false, "Test");
DEFINE_int32(channels, 3, "Image channels");

void load_session(DeepFlow *df, std::string prefix, std::string suffix) {
	std::string filename = prefix + suffix + ".bin";
	std::cout << "Loading " << prefix << " from " << filename << std::endl;
	df->block()->load_from_binary(filename);
}

void create_graph(DeepFlow *df) {	

	auto solver = df->adam_solver(0.002f, 0.5f, 0.99f);
	auto negative_slope = 0.2;	
	auto fn = 128;
	auto ef = 0.1f;

	df->image_batch_reader(FLAGS_faces, { FLAGS_batch, FLAGS_channels, 128, 128 }, true, "face_data");
	auto loss_input = df->place_holder({ FLAGS_batch, FLAGS_channels, 128, 128 }, Tensor::Float, "loss_input");
	auto loss_target = df->place_holder({ FLAGS_batch, FLAGS_channels, 128, 128 }, Tensor::Float, "loss_target");
	auto sqerr = df->square_error(loss_input, loss_target);
	auto loss = df->loss(sqerr, DeepFlow::AVG);
	auto disp_input = df->place_holder({ FLAGS_batch, FLAGS_channels, 128, 128 }, Tensor::Float, "disp_input");
	df->display(disp_input, 1, DeepFlow::VALUES, 1, "disp");

	auto net = df->place_holder({ FLAGS_batch, FLAGS_channels, 128, 128 }, Tensor::Float, "enc_input");
	
	net = df->conv2d(net, FLAGS_channels, fn, 3, 1, 2, true, solver, "conv1");
	net = df->leaky_relu(net, negative_slope);	

	net = df->conv2d(net, fn, fn, 3, 1, 2, true, solver, "conv2");	
	net = df->leaky_relu(net, negative_slope);	
	
	net = df->conv2d(net, fn, fn, 3, 1, 2, true, solver, "conv3");
	net = df->leaky_relu(net, negative_slope);	

	net = df->conv2d(net, fn, fn, 3, 1, 2, true, solver, "conv4");
	net = df->leaky_relu(net, negative_slope);

	net = df->conv2d(net, fn, fn, 3, 1, 2, false, solver, "conv5");	
	net = df->leaky_relu(net, negative_slope);

	net = df->transposed_conv2d(net, fn, fn, 3, 1, 2, false, solver, "tconv1");
	net = df->batch_normalization(net, solver, fn, ef, "tconv1_bn");
	net = df->leaky_relu(net, negative_slope);

	net = df->transposed_conv2d(net, fn, fn , 3, 1, 2, false, solver, "tconv2");
	net = df->batch_normalization(net, solver, fn, ef, "tconv2_bn");
	net = df->leaky_relu(net, negative_slope);
	
	net = df->transposed_conv2d(net, fn, fn, 3, 1, 2, false, solver, "tconv3");
	net = df->batch_normalization(net, solver, fn, ef, "tconv3_bn");
	net = df->leaky_relu(net, negative_slope);

	net = df->transposed_conv2d(net, fn, fn, 3, 1, 2, false, solver, "tconv4");
	net = df->batch_normalization(net, solver, fn, ef, "tconv4_bn");
	net = df->leaky_relu(net, negative_slope);	

	net = df->transposed_conv2d(net, fn, 3, 3, 1, 2, true, solver, "tconv5");		

	df->tanh(net, "dec_output");

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
		load_session(&df, "face_ac", FLAGS_load);

	std::shared_ptr<Session> session = df.session();
	session->initialize(execution_context);

	auto face_data = session->get_node("face_data");
	auto enc_input = session->get_placeholder("enc_input");
	auto dec_output = session->get_node("dec_output");
	auto disp_input = session->get_placeholder("disp_input");
	auto disp = session->get_node("disp");
	auto loss_input = session->get_placeholder("loss_input");
	auto loss_target = session->get_placeholder("loss_target");
	auto loss = session->get_node<Loss>("loss");

	loss->set_alpha(0.0001f);
	loss->set_beta(1 - 0.0001f);

	if (FLAGS_train) {
		int iter;
		for (iter = 1; iter <= FLAGS_iter && execution_context->quit != true; ++iter) {

			execution_context->execution_mode = ExecutionContext::TRAIN;
			execution_context->current_epoch = iter;
			execution_context->current_iteration = iter;

			session->forward({ face_data });
			session->forward({ dec_output }, { { enc_input, face_data->output(0)->value() } });
			session->forward({ loss }, {
				{ loss_input , dec_output->output(0)->value() },
				{ loss_target , face_data->output(0)->value() }
			});
			session->backward({ loss });
			session->backward({ dec_output }, { { dec_output , loss_input->output(0)->diff() } });
			session->apply_solvers();

			execution_context->execution_mode = ExecutionContext::TEST;
			session->forward({ face_data });
			session->forward({ dec_output }, { { enc_input, face_data->output(0)->value() } });
			session->forward({ disp }, { { disp_input , dec_output->output(0)->value() } });

			if (FLAGS_save != 0 && iter % FLAGS_save == 0) {
				session->save("face_ac" + std::to_string(iter) + ".bin");
			}

		}

		if (FLAGS_save != 0) {
			session->save("face_ac" + std::to_string(iter) + ".bin");
		}
	}

	if (FLAGS_test) {
		execution_context->execution_mode = ExecutionContext::TEST;
		int iter;
		for (iter = 1; iter <= FLAGS_iter && execution_context->quit != true; ++iter) {
			session->forward({ face_data });
			session->forward({ dec_output }, { { enc_input, face_data->output(0)->value() } });
			session->forward({ disp }, { { disp_input , dec_output->output(0)->value() } });
		}
	}
	
	cudaDeviceReset();
}

