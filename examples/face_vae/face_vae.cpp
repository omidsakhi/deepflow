
#include "core/deep_flow.h"
#include "core/session.h"

#include <random>
#include <gflags/gflags.h>

DEFINE_string(faces, "C:/Projects/deepflow/data/celeba128", "Path to face 128x128 dataset folder");
DEFINE_int32(batch, 40, "Batch size");
DEFINE_int32(debug, 0, "Level of debug");
DEFINE_int32(iter, 2000000, "Maximum iterations");
DEFINE_string(load, "", "Load from face_acXXXX.bin");
DEFINE_int32(save, 1000 , "Save model iter frequency (Don't Save = 0)");
DEFINE_bool(train, false, "Train");
DEFINE_bool(test, false, "Test");
DEFINE_int32(channels, 3, "Image channels");
DEFINE_int32(bottleneck, 1024, "Network bottleneck");

void load_session(DeepFlow *df, std::string prefix, std::string suffix) {
	std::string filename = prefix + suffix + ".bin";
	std::cout << "Loading " << prefix << " from " << filename << std::endl;
	df->block()->load_from_binary(filename);
}

void create_graph(DeepFlow *df) {	

	auto solver = df->adam_solver( AdamSolverOp().lr(0.002f).beta1(0.5f).beta2(0.9f));

	auto ns = 0.2f;	
	auto fn = 128;	

	df->imbatch(FLAGS_faces, { FLAGS_batch, FLAGS_channels, 128, 128 }, ImbatchOp("face_data"));
	auto loss_input = df->place_holder({ FLAGS_batch, FLAGS_channels, 128, 128 }, PlaceholderOp("loss_input"));
	auto loss_target = df->place_holder({ FLAGS_batch, FLAGS_channels, 128, 128 }, PlaceholderOp("loss_target"));
	auto sqerr = df->square_error(loss_input, loss_target);
	auto loss = df->loss(sqerr);
	auto disp_input = df->place_holder({ FLAGS_batch, FLAGS_channels, 128, 128 }, PlaceholderOp("disp_input"));
	df->display(disp_input, DisplayOp("disp").iter(100));

	df->with("encoder");

	auto net = df->place_holder({ FLAGS_batch, FLAGS_channels, 128, 128 }, PlaceholderOp("enc_input"));
	
	net = df->conv2d(net, FLAGS_channels, fn, solver, ConvolutionOp("e64").kernel(3).pad(1).stride(2).with_bias());
	net = df->leaky_relu(net, LeakyReluOp().negative_slope(ns));	

	net = df->conv2d(net, fn, fn, solver, ConvolutionOp("e32").kernel(3).pad(1).stride(2).with_bias());
	net = df->lrn(net, LrnOp().n(3));
	net = df->leaky_relu(net, LeakyReluOp().negative_slope(ns));
	
	net = df->conv2d(net, fn, fn, solver, ConvolutionOp("e16").kernel(3).pad(1).stride(2).with_bias());	
	net = df->lrn(net, LrnOp().n(3));
	net = df->leaky_relu(net, LeakyReluOp().negative_slope(ns));

	net = df->conv2d(net, fn, fn, solver, ConvolutionOp("e8").kernel(3).pad(1).stride(2).with_bias());	
	net = df->lrn(net, LrnOp().n(3));
	net = df->leaky_relu(net, LeakyReluOp().negative_slope(ns));

	net = df->conv2d(net, fn, fn, solver, ConvolutionOp("e4").kernel(3).pad(1).stride(2).with_bias());
	net = df->lrn(net, LrnOp().n(3));
	net = df->leaky_relu(net, LeakyReluOp().negative_slope(ns));

	auto mean = df->dense(net, { fn * 4 * 4, FLAGS_bottleneck, 1, 1 }, solver, DenseOp("mean"));
	auto sigma = df->dense(net, { fn * 4 * 4, FLAGS_bottleneck, 1, 1 }, solver, DenseOp("sigma"));

	sigma = df->sigmoid(sigma);

	net = df->gaussian(mean, sigma);

	df->with("decoder");

	net = df->dense(net, { FLAGS_bottleneck, fn, 4, 4 }, solver, DenseOp("d_input"));

	net = df->transposed_conv2d(net, fn, fn, solver, ConvolutionOp("d8").kernel(3).pad(1).stride(2).with_bias());
	net = df->lrn(net, LrnOp().n(3));	
	net = df->leaky_relu(net, LeakyReluOp().negative_slope(ns));

	net = df->transposed_conv2d(net, fn, fn, solver, ConvolutionOp("d16").kernel(3).pad(1).stride(2).with_bias());
	net = df->lrn(net, LrnOp().n(3));	
	net = df->leaky_relu(net, LeakyReluOp().negative_slope(ns));
	
	net = df->transposed_conv2d(net, fn, fn, solver, ConvolutionOp("d32").kernel(3).pad(1).stride(2).with_bias());	
	net = df->lrn(net, LrnOp().n(3));	
	net = df->leaky_relu(net, LeakyReluOp().negative_slope(ns));

	net = df->transposed_conv2d(net, fn, fn, solver, ConvolutionOp("d64").kernel(3).pad(1).stride(2).with_bias());		
	net = df->lrn(net, LrnOp().n(3));
	net = df->leaky_relu(net, LeakyReluOp().negative_slope(ns));

	net = df->transposed_conv2d(net, fn, 3, solver, ConvolutionOp("dec_output").kernel(3).pad(1).stride(2). with_bias());		

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
		load_session(&df, "face_vae", FLAGS_load);

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
				session->save("face_vae" + std::to_string(iter) + ".bin");
			}

		}

		if (FLAGS_save != 0) {
			session->save("face_vae" + std::to_string(iter) + ".bin");
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

