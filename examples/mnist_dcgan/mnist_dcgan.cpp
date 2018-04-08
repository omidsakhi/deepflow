
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
	int fn = 64;
	float ef = 0.01f;
	auto solver = df->adam_solver(0.00005f, 0.5f, 0.99f, 10e-8, "g_adam");

	auto node = df->place_holder({ FLAGS_batch, FLAGS_z_dim, 1, 1 }, Tensor::Float, "g_input");

	node = df->dense(node, { FLAGS_z_dim, fn * 4, 4 , 4 }, false, solver, "gfc");
	node = df->batch_normalization(node, solver, fn * 4, ef, "gfc_bn");
	node = df->relu(node);

	node = df->transposed_conv2d(node, fn * 4, fn * 4, 3, 1, 2, false, solver, "g8");
	node = df->batch_normalization(node, solver, fn * 4, ef, "g8_bn");
	node = df->relu(node);

	node = df->transposed_conv2d(node, fn * 4, fn * 4, 3, 1, 2, false, solver, "g16");
	node = df->batch_normalization(node, solver, fn * 4, ef, "g16_bn");
	node = df->relu(node);

	node = df->transposed_conv2d(node, fn * 4, fn * 2, 3, 1, 2, false, solver, "g32");
	node = df->batch_normalization(node, solver, fn * 2, ef, "g32_bn");
	node = df->relu(node);

	df->conv2d(node, fn * 2, 1, 5, 0, 1, true, solver, "g28");

}

void create_discriminator(DeepFlow *df) {
	int fn = 64;
	float ef = 0.01f;
	auto solver = df->adam_solver(0.00005f, 0.5f, 0.99f, 10e-8, "d_adam");

	auto node = df->place_holder({ FLAGS_batch , 1, 28, 28 }, Tensor::Float, "d_input");

	node = df->conv2d(node, 1, fn, 5, 2, 2, false, solver, "d14");
	node = df->batch_normalization(node, solver, fn, ef, "d14_bn");
	node = df->leaky_relu(node, 0.2f);

	node = df->conv2d(node, fn, fn * 2, 5, 2, 2, false, solver, "d7");
	node = df->batch_normalization(node, solver, fn * 2, ef, "d7_bn");
	node = df->leaky_relu(node, 0.2f);

	node = df->conv2d(node, fn * 2, fn * 4, 5, 2, 2, false, solver, "d4");
	node = df->batch_normalization(node, solver, fn * 4, ef, "d4_bn");
	node = df->leaky_relu(node, 0.2f);

	df->dense(node, { (fn * 4) * 4 * 4, 1, 1, 1 }, true, solver, "dfc");

}

void main(int argc, char** argv) {

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();

	auto execution_context = std::make_shared<ExecutionContext>();
	execution_context->debug_level = FLAGS_debug;

	DeepFlow df;

	if (FLAGS_load.empty()) {
		df.data_generator(df.fill({ FLAGS_batch, 1, 1, 1 }, 1), "", "mnist_labels");
		df.mnist_reader(FLAGS_mnist, FLAGS_batch, MNISTReader::MNISTReaderType::Train, MNISTReader::Data, "mnist_data");
		df.data_generator(df.fill({ FLAGS_batch, 1, 1, 1 }, 0), "", "generator_labels");
		df.data_generator(df.random_uniform({ FLAGS_batch, FLAGS_z_dim, 1, 1 }, -1, 1), "", "z");
		df.data_generator(df.random_uniform({ FLAGS_batch, FLAGS_z_dim, 1, 1 }, -1, 1), "", "static_z");
		auto imwrite_input = df.place_holder({ FLAGS_batch, 1, 28, 28 }, Tensor::Float, "imwrite_input");
		df.imwrite(imwrite_input, "{it}", true, "imwrite");
		auto discriminator_input = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "loss_input");
		auto discriminator_target = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "loss_target");
		std::string err = df.reduce_mean(df.square_error(discriminator_input, discriminator_target));
		df.loss(err, DeepFlow::AVG, 1.0f, 0.0f, "loss");
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
	auto generator_output = session->get_node("g28");
	auto generator_input = session->get_placeholder("g_input");
	auto discriminator_input = session->get_placeholder("d_input");
	auto discriminator_output = session->get_node("dfc_bias");
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

			session->apply_solvers({ "d_adam" });

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

			session->apply_solvers({ "g_adam" });
			session->reset_gradients();

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