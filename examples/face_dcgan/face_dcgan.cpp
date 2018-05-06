
#include "core/deep_flow.h"
#include "core/session.h"
#include "utilities/moving_average.h"

#include <random>
#include <gflags/gflags.h>

DEFINE_string(faces, "C:/Projects/deepflow/data/celeba128", "Path to face dataset folder");
DEFINE_int32(channels, 3, "Image channels");
DEFINE_int32(size, 128, "Image width/height");
DEFINE_int32(z_dim, 100, "Z dimention");
DEFINE_int32(total, 200000, "Total images");
DEFINE_int32(batch, 40, "Batch size");
DEFINE_int32(debug, 0, "Debug Level");
DEFINE_int32(iter, 2000000, "Maximum iterations");
DEFINE_string(load, "", "Load from face_dcganXXXX.bin");
DEFINE_int32(save_image, 100, "Save image iteration frequency (Don't Save = 0)");
DEFINE_int32(save_model, 1000, "Save model iteration frequency (Don't Save = 0)");
DEFINE_int32(print, 0, "Print source C++");
DEFINE_bool(train, false, "Train");
DEFINE_bool(test, false, "Test");

void load_session(DeepFlow *df, std::string prefix) {
	std::string filename = prefix + ".bin";
	std::cout << "Loading from " << filename << std::endl;	
	df->block()->load_from_binary(filename);	
}

void create_loss(DeepFlow *df, int num) {
	
	df->with("loss");

	auto discriminator_input = df->place_holder({ FLAGS_batch, 1, 1, 1 }, PlaceholderOp("loss" + std::to_string(num) + "_input"));	
	auto discriminator_target = df->place_holder({ FLAGS_batch, 1, 1, 1 }, PlaceholderOp("loss" + std::to_string(num) + "_target"));
	std::string err = df->square_error(discriminator_input, discriminator_target);
	df->loss(err, LossOp("loss" + std::to_string(num)));
}

void create_generator(DeepFlow *df) {	
	
	df->with("generator");

	int fn = 64;
	float ef = 0.01f;

	auto solver = df->adam_solver( AdamSolverOp("g_adam").lr(0.0002f).beta1(0.5f).beta2(0.98f));
	
	auto node = df->place_holder({ FLAGS_batch, FLAGS_z_dim, 1, 1 }, PlaceholderOp("g_input"));

	node = df->dense(node, { FLAGS_z_dim, fn * 4, 4 , 4 }, solver, DenseOp("gfc").no_bias());
	node = df->batch_normalization(node, fn * 4, solver, BatchNormalizationOp("gfc_bn").exponent_factor(ef));	
	node = df->relu(node);

	node = df->transposed_conv2d(node, fn * 4, fn * 4, solver, ConvolutionOp("g8_1").kernel(3).stride(2));
	node = df->batch_normalization(node, fn * 4, solver, BatchNormalizationOp("g8_1_bn").exponent_factor(ef));
	node = df->relu(node);

	node = df->transposed_conv2d(node, fn * 4, fn * 4, solver, ConvolutionOp("g8_2").kernel(3).stride(1));
	node = df->batch_normalization(node, fn * 4, solver, BatchNormalizationOp("g8_2_bn").exponent_factor(ef));
	auto g8 = node;
	node = df->relu(node);

	node = df->transposed_conv2d(node, fn * 4, fn * 4, solver, ConvolutionOp("g16_1").kernel(3).stride(2));
	node = df->batch_normalization(node, fn * 4, solver, BatchNormalizationOp("g16_1_bn").exponent_factor(ef));
	node = df->relu(node);

	node = df->transposed_conv2d(node, fn * 4, fn * 4, solver, ConvolutionOp("g16_2").kernel(3).stride(1));
	node = df->batch_normalization(node, fn * 4, solver, BatchNormalizationOp("g16_2_bn").exponent_factor(ef));
	auto g16 = node;
	node = df->relu(node);

	node = df->transposed_conv2d(node, fn * 4, fn * 4, solver, ConvolutionOp("g32_1").kernel(3).stride(2));
	node = df->batch_normalization(node, fn * 4, solver, BatchNormalizationOp("g32_1_bn").exponent_factor(ef));
	node = df->relu(node);

	node = df->transposed_conv2d(node, fn * 4, fn * 4, solver, ConvolutionOp("g32_2").kernel(3).stride(1));
	node = df->batch_normalization(node, fn * 4, solver, BatchNormalizationOp("g32_2_bn").exponent_factor(ef));
	auto g32 = node;
	node = df->relu(node);

	node = df->lifting(node, LiftingOp().up()); // 64x64

	node = df->transposed_conv2d(node, fn, fn * 4, solver, ConvolutionOp("g64").kernel(3).stride(1));
	node = df->batch_normalization(node, fn * 4, solver, BatchNormalizationOp("g64_bn").exponent_factor(ef));
	node = df->relu(node);

	node = df->lifting(node, LiftingOp().up()); // 128x128

	node = df->transposed_conv2d(node, fn, fn * 4, solver, ConvolutionOp("g128_1").kernel(3).stride(1));
	node = df->batch_normalization(node, fn * 4, solver, BatchNormalizationOp("g128_1_bn").exponent_factor(ef));
	node = df->add(df->resize(g8, 16, 16), node);
	node = df->add(df->resize(g16, 8, 8), node);
	node = df->add(df->resize(g32, 4, 4), node);
	node = df->relu(node);

	node = df->transposed_conv2d(node, fn * 4, 3, solver, ConvolutionOp("g128_2").kernel(3).stride(1));
	
}

void create_discriminator(DeepFlow *df) {
	
	df->with("discriminator");
	
	int fn = 64;	

	auto solver = df->adam_solver( AdamSolverOp("d_adam").lr(0.0002f).beta1(0.5f).beta2(0.98f));	

	auto node = df->place_holder({ FLAGS_batch , FLAGS_channels, FLAGS_size, FLAGS_size }, PlaceholderOp("d_input"));	

	node = df->lifting(node, LiftingOp().down().with_flip()); // 64x64

	node = df->conv2d(node, 3 * 4, fn, solver, ConvolutionOp("d641").kernel(3).stride(1).with_bias());	
	node = df->leaky_relu(node);

	node = df->lifting(node, LiftingOp().down().with_flip()); // 32x32

	node = df->conv2d(node, fn * 4, fn * 2, solver, ConvolutionOp("d321").kernel(3).stride(1).with_bias());
	node = df->leaky_relu(node);

	node = df->lifting(node, LiftingOp().down().with_flip()); // 16x16

	node = df->conv2d(node, fn * 8, fn * 2, solver, ConvolutionOp("d16").kernel(3).stride(1).with_bias());
	node = df->leaky_relu(node);

	node = df->lifting(node, LiftingOp().down().with_flip()); // 8x8

	node = df->conv2d(node, fn * 8, fn * 4, solver, ConvolutionOp("d6").kernel(3).pad(0).stride(1).with_bias());
	node = df->leaky_relu(node);

	node = df->conv2d(node, fn * 4, fn * 4, solver, ConvolutionOp("d4").kernel(3).pad(0).stride(1).with_bias());
	node = df->leaky_relu(node);

	node = df->dense(node, { (fn * 4) * 4 * 4, 1, 1, 1 }, solver, DenseOp("ddense").with_bias());	

	df->sigmoid(node, SigmoidOp("d_output"));
}

void main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();

	auto execution_context = std::make_shared<ExecutionContext>();
	execution_context->debug_level = FLAGS_debug;

	DeepFlow df;	

	if (FLAGS_load.empty()) {
		df.imbatch(FLAGS_faces, { FLAGS_batch, FLAGS_channels, FLAGS_size, FLAGS_size }, ImbatchOp("face_data"));
		df.data_generator(df.fill({ FLAGS_batch, 1, 1, 1 }, 1), DataGeneratorOp("face_labels"));
		df.data_generator(df.fill({ FLAGS_batch, 1, 1, 1 }, 0), DataGeneratorOp("generator_labels"));
		df.data_generator(df.random_uniform({ FLAGS_batch, FLAGS_z_dim, 1, 1 }, -1, 1), DataGeneratorOp("z"));
		df.data_generator(df.random_uniform({ FLAGS_batch, FLAGS_z_dim, 1, 1 }, -1, 1), DataGeneratorOp("static_z"));
		auto imwrite_input = df.place_holder({ FLAGS_batch, FLAGS_channels, FLAGS_size, FLAGS_size }, PlaceholderOp("imwrite_input"));
		df.imwrite(imwrite_input, "{it}");
		create_loss(&df, 1);
		create_loss(&df, 2);
		create_discriminator(&df);
		create_generator(&df);
	}
	else {
		load_session(&df, "face_dcgan" + FLAGS_load);
	}

	auto session = df.session();
	session->initialize(execution_context);	

	if (FLAGS_print) {
		std::cout << session->to_cpp() << std::endl;		
		return;
	}

	auto face_data = session->get_node("face_data");	
	auto generator_output = session->end_node("generator");
	auto generator_input = session->get_placeholder("g_input");
	auto discriminator_input = session->get_placeholder("d_input");	
	auto discriminator_output = session->end_node("discriminator");
	auto generator_labels = session->get_node("generator_labels");
	auto face_labels = session->get_node("face_labels");
	auto loss1_input = session->get_placeholder("loss1_input");	
	auto loss1_target = session->get_placeholder("loss1_target");
	auto loss2_input = session->get_placeholder("loss2_input");
	auto loss2_target = session->get_placeholder("loss2_target");
	auto loss1 = session->get_node<Loss>("loss1");
	auto loss2 = session->get_node<Loss>("loss2");
	auto imwrite_input = session->get_placeholder("imwrite_input");
	auto imwrite = session->get_node<ImageWriter>("imwrite");
	auto z = session->get_node("z");
	auto static_z = session->get_node("static_z");

	int iter = 1;
		
	if (FLAGS_train) {
		static_z->forward();

		MovingAverage g_loss_avg(1000);
		MovingAverage p_d_loss_avg(1000);
		MovingAverage n_d_loss_avg(1000);

		if (!FLAGS_load.empty())
			iter = std::stoi(FLAGS_load) + 1;

		session->forward({ face_labels });
		session->forward({ generator_labels });

		for (; iter <= FLAGS_iter && execution_context->quit != true; ++iter) {

			execution_context->current_iteration = iter;
			std::cout << "Iteration: [" << iter << "/" << FLAGS_iter << "]";

			session->forward({ face_data });
			session->forward({ discriminator_output }, { { discriminator_input, face_data->output(0)->value() } });
			session->forward({ loss1 }, {
				{ loss1_input , discriminator_output->output(0)->value() } ,
				{ loss1_target , face_labels->output(0)->value() } ,
			});
			p_d_loss_avg.add(loss1->output(0)->value()->to_float());
			std::cout << " - p_d_loss: " << p_d_loss_avg.result();
			session->backward({ loss1 });
			session->backward({ discriminator_output }, { { discriminator_output , loss1_input->output(0)->diff() } });

			session->forward({ z });
			session->forward({ generator_output }, { { generator_input , z->output(0)->value() } });
			session->forward({ discriminator_output }, { { discriminator_input, generator_output->output(0)->value() } });
			session->forward({ loss1 }, {
				{ loss1_input , discriminator_output->output(0)->value() },
				{ loss1_target, generator_labels->output(0)->value() }
			});
			n_d_loss_avg.add(loss1->output(0)->value()->to_float());
			std::cout << " - n_d_loss: " << n_d_loss_avg.result();
			session->backward({ loss1 });
			session->backward({ discriminator_output }, { { discriminator_output, loss1_input->output(0)->diff() } });

			session->apply_solvers("discriminator");

			session->forward({ discriminator_output }, { { discriminator_input, generator_output->output(0)->value() } });
			session->forward({ loss2 }, {
				{ loss2_input , discriminator_output->output(0)->value() },
				{ loss2_target, face_labels->output(0)->value() }
			});
			float g_loss = loss2->output(0)->value()->to_float();
			g_loss_avg.add(g_loss);
			session->backward({ loss2 });
			session->backward({ discriminator_output }, { { discriminator_output, loss2_input->output(0)->diff() } });
			session->backward({ generator_output }, { { generator_output, discriminator_input->output(0)->diff() } });
			std::cout << " - g_loss: " << g_loss_avg.result() << std::endl;

			session->apply_solvers("generator");
			session->reset_gradients("discriminator");

			if (FLAGS_save_image != 0 && iter % FLAGS_save_image == 0) {
				imwrite->set_text_line(
					"Iteration: " + std::to_string(iter) +
					"  g " + std::to_string(g_loss_avg.result()) +
					" +d " + std::to_string(p_d_loss_avg.result()) +
					" -d " + std::to_string(n_d_loss_avg.result()));
				session->forward({ generator_output }, { { generator_input, static_z->output(0)->value() } });
				session->forward({ imwrite }, { { imwrite_input, generator_output->output(0)->value() } });
			}

			if (FLAGS_save_model != 0 && iter % FLAGS_save_model == 0) {
				session->save("face_dcgan" + std::to_string(iter) + ".bin");
			}
		}
	}
	
	if (FLAGS_test) {
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

