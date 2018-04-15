
#include "core/deep_flow.h"
#include "core/session.h"
#include "utilities/moving_average.h"

#include <random>
#include <gflags/gflags.h>

DEFINE_string(mnist, "C:/Projects/deepflow/data/mnist", "Path to mnist folder dataset");
DEFINE_int32(batch, 100, "Batch size");
DEFINE_int32(z_dim, 100, "Z dimention");
DEFINE_int32(debug, 0, "Level of debug");
DEFINE_int32(iter, 1000000, "Maximum iterations");
DEFINE_string(load, "", "Load from mnist_dcganXXXX.bin");
DEFINE_int32(save_image, 100, "Save image iteration frequency (Don't Save = 0)");
DEFINE_int32(save_model, 1000, "Save model iteration frequency (Don't Save = 0)");
DEFINE_bool(train, false, "Train");
DEFINE_bool(test, false, "Test");
DEFINE_bool(cpp, false, "Print C++ code");
DEFINE_bool(print, false, "Print source C++");

void load_session(DeepFlow *df, std::string prefix) {
	std::string filename = prefix + ".bin";
	std::cout << "Loading " << prefix << " from " << filename << std::endl;
	df->block()->load_from_binary(filename);
}

void create_generator(DeepFlow *df) {
	df->with("generator");

	int fn = 64;
	float ef = 0.01f;
	auto solver = df->adam_solver( AdamSolverOp("g_adam").lr(0.00005f).beta1(0.5f).beta2(0.99f) );

	auto node = df->place_holder({ FLAGS_batch, FLAGS_z_dim, 1, 1 }, PlaceholderOp("g_input"));

	node = df->dense(node, { FLAGS_z_dim, fn * 4, 4 , 4 }, solver, DenseOp("gfc"));
	node = df->batch_normalization(node, fn * 4, solver, BatchNormalizationOp("gfc_bn").exponent_factor(ef));
	node = df->relu(node);

	node = df->transposed_conv2d(node, fn * 4, fn * 4, solver, ConvolutionOp("g8").kernel(3).pad(1).stride(2));
	node = df->batch_normalization(node, fn * 4, solver, BatchNormalizationOp("g8_bn").exponent_factor(ef));
	node = df->relu(node);

	node = df->transposed_conv2d(node, fn * 4, fn * 4, solver, ConvolutionOp("g16").kernel(3).pad(1).stride(2));
	node = df->batch_normalization(node, fn * 4, solver, BatchNormalizationOp("g16_bn").exponent_factor(ef));
	node = df->relu(node);

	node = df->transposed_conv2d(node, fn * 4, fn * 2, solver, ConvolutionOp("g32").kernel(3).pad(1).stride(2));
	node = df->batch_normalization(node, fn * 2, solver, BatchNormalizationOp("g32_bn").exponent_factor(ef));
	node = df->relu(node);

	df->conv2d(node, fn * 2, 1, solver, ConvolutionOp("g_output").kernel(5).pad(0).stride(1).with_bias());

}

void create_discriminator(DeepFlow *df) {
	df->with("discriminator");

	int fn = 64;
	float ef = 0.01f;
	auto solver = df->adam_solver(AdamSolverOp("d_adam").lr(0.00005f).beta1(0.5f).beta2(0.99f));

	auto node = df->place_holder({ FLAGS_batch , 1, 28, 28 }, PlaceholderOp("d_input"));

	node = df->conv2d(node, 1, fn, solver, ConvolutionOp("d14").kernel(5).pad(2).stride(2));
	node = df->batch_normalization(node, fn, solver, BatchNormalizationOp("g14_bn").exponent_factor(ef));
	node = df->leaky_relu(node);

	node = df->conv2d(node, fn, fn * 2, solver, ConvolutionOp("d7").kernel(5).pad(2).stride(2));
	node = df->batch_normalization(node, fn * 2, solver, BatchNormalizationOp("g7_bn").exponent_factor(ef));
	node = df->leaky_relu(node);

	node = df->conv2d(node, fn * 2, fn * 4, solver, ConvolutionOp("d4").kernel(5).pad(2).stride(2));
	node = df->batch_normalization(node, fn * 4, solver, BatchNormalizationOp("g4_bn").exponent_factor(ef));
	node = df->leaky_relu(node);

	df->dense(node, { (fn * 4) * 4 * 4, 1, 1, 1 }, solver, DenseOp("d_output"));

}

void main(int argc, char** argv) {

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();

	auto execution_context = std::make_shared<ExecutionContext>();
	execution_context->debug_level = FLAGS_debug;

	DeepFlow df;

	if (FLAGS_load.empty()) {		
		df.mnist_reader(FLAGS_mnist, MNISTReaderOp("mnist_data").batch(FLAGS_batch).train().data());
		df.data_generator(df.fill({ FLAGS_batch, 1, 1, 1 }, 1), DataGeneratorOp("mnist_labels"));
		df.data_generator(df.fill({ FLAGS_batch, 1, 1, 1 }, 0), DataGeneratorOp("generator_labels"));
		df.data_generator(df.random_uniform({ FLAGS_batch, FLAGS_z_dim, 1, 1 }, -1, 1), DataGeneratorOp("z"));
		df.data_generator(df.random_uniform({ FLAGS_batch, FLAGS_z_dim, 1, 1 }, -1, 1), DataGeneratorOp("static_z"));
		auto imwrite_input = df.place_holder({ FLAGS_batch, 1, 28, 28 }, PlaceholderOp("imwrite_input"));
		df.imwrite(imwrite_input, "{it}");
		auto discriminator_input = df.place_holder({ FLAGS_batch, 1, 1, 1 }, PlaceholderOp("loss_input"));
		auto discriminator_target = df.place_holder({ FLAGS_batch, 1, 1, 1 }, PlaceholderOp("loss_target"));
		std::string err = df.reduce_all(df.square_error(discriminator_input, discriminator_target));
		df.loss(err);
		create_discriminator(&df);
		create_generator(&df);
	}
	else {
		load_session(&df, "minst_dcgan" + FLAGS_load);
	}

	auto session = df.session();
	session->initialize(execution_context);

	if (FLAGS_print) {
		std::cout << session->to_cpp() << std::endl;
		return;
	}

	auto mnist_data = session->get_node("mnist_data");
	auto generator_output = session->get_node("g_output");
	auto generator_input = session->get_placeholder("g_input");
	auto discriminator_input = session->get_placeholder("d_input");
	auto discriminator_output = session->get_node("d_output");
	auto generator_labels = session->get_node("generator_labels");
	auto mnist_labels = session->get_node("mnist_labels");
	auto loss_input = session->get_placeholder("loss_input");
	auto loss_target = session->get_placeholder("loss_target");
	auto loss = session->get_node<Loss>("loss");
	auto imwrite_input = session->get_placeholder("imwrite_input");
	auto imwrite = session->get_node("imwrite");
	auto z = session->get_node("z");
	auto static_z = session->get_node("static_z");

	int iter = 0;

	if (FLAGS_train) {
		static_z->forward();

		MovingAverage g_loss_avg(100);
		MovingAverage p_d_loss_avg(100);
		MovingAverage n_d_loss_avg(100);

		if (FLAGS_load.empty()) {
			iter = 1;
		}
		else {
			iter = std::stoi(FLAGS_load) + 1;
		}		

		session->forward({ mnist_labels });
		session->forward({ generator_labels });

		for (; iter <= FLAGS_iter && execution_context->quit != true; ++iter) {

			execution_context->current_iteration = iter;
			std::cout << "Iteration: [" << iter << "/" << FLAGS_iter << "]";

			session->forward({ mnist_data });
			session->forward({ discriminator_output }, { { discriminator_input, mnist_data->output(0)->value() } });
			session->forward({ loss }, {
				{ loss_input , discriminator_output->output(0)->value() } ,
				{ loss_target , mnist_labels->output(0)->value() } ,
			});
			p_d_loss_avg.add(loss->output(0)->value()->toFloat());
			std::cout << " - p_d_loss: " << p_d_loss_avg.result();
			session->backward({ loss });
			session->backward({ discriminator_output }, { { discriminator_output , loss_input->output(0)->diff() } });

			session->forward({ z });
			session->forward({ generator_output }, { { generator_input , z->output(0)->value() } });
			session->forward({ discriminator_output }, { { discriminator_input, generator_output->output(0)->value() } });
			session->forward({ loss }, {
				{ loss_input , discriminator_output->output(0)->value() },
				{ loss_target, generator_labels->output(0)->value() }
			});
			n_d_loss_avg.add(loss->output(0)->value()->toFloat());
			std::cout << " - n_d_loss: " << n_d_loss_avg.result();
			session->backward({ loss });
			session->backward({ discriminator_output }, { { discriminator_output, loss_input->output(0)->diff() } });

			session->apply_solvers("discriminator");

			session->forward({ discriminator_output }, { { discriminator_input, generator_output->output(0)->value() } });
			session->forward({ loss }, {
				{ loss_input , discriminator_output->output(0)->value() },
				{ loss_target, mnist_labels->output(0)->value() }
			});
			float g_loss = loss->output(0)->value()->toFloat();
			g_loss_avg.add(g_loss);
			session->backward({ loss });
			session->backward({ discriminator_output }, { { discriminator_output, loss_input->output(0)->diff() } });
			session->backward({ generator_output }, { { generator_output, discriminator_input->output(0)->diff() } });
			std::cout << " - g_loss: " << g_loss_avg.result() << std::endl;

			session->apply_solvers("generator");
			session->reset_gradients("discriminator");

			if (FLAGS_save_image != 0 && iter % FLAGS_save_image == 0) {
				session->forward({ generator_output }, { { generator_input, static_z->output(0)->value() } });
				session->forward({ imwrite }, { { imwrite_input, generator_output->output(0)->value() } });
			}

			if (FLAGS_save_model != 0 && iter % FLAGS_save_model == 0) {
				session->save("mnist_dcgan" + std::to_string(iter) + ".bin");
			}
		}
	}

	if (FLAGS_test) {
		iter = 1;
		for (; iter <= FLAGS_iter && execution_context->quit != true; ++iter) {

			execution_context->current_iteration = iter;
			std::cout << "Iteration: [" << iter << "/" << FLAGS_iter << "]\n";
			session->forward({ z });
			session->forward({ generator_output }, { { generator_input , z->output(0)->value() } });
			session->forward({ imwrite }, { { imwrite_input, generator_output->output(0)->value() } });
		}
	}

	cudaDeviceReset();
}