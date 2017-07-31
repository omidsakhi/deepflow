
#include "core/deep_flow.h"
#include "core/session.h"

#include <random>
#include <gflags/gflags.h>

DEFINE_string(faces, "C:/Projects/deepflow/data/face643", "Path to face 64x64 dataset folder");
DEFINE_string(i, "", "Trained network model to load");
DEFINE_string(o, "", "Trained network model to save");
DEFINE_bool(text, false, "Save model as text");
DEFINE_bool(includeweights, false, "Also save weights in text mode");
DEFINE_bool(includeinits, false, "Also save initial values");
DEFINE_int32(batch, 49, "Batch size");
DEFINE_string(run,"", "Phase to execute graph");
DEFINE_bool(printiter, false, "Print iteration message");
DEFINE_bool(printepoch, true, "Print epoch message");
DEFINE_int32(debug, 0, "Level of debug");
DEFINE_int32(epoch, 1000000, "Maximum epochs");
DEFINE_int32(iter, -1, "Maximum iterations");
DEFINE_int32(channels, 3, "Image channels");
DEFINE_int32(total, 13230, "Image channels");
DEFINE_bool(cpp, false, "Print C++ code");

std::shared_ptr<Session> create_face_reader() {
	DeepFlow df;
	df.image_batch_reader(FLAGS_faces, { FLAGS_batch, FLAGS_channels, 64, 64 }, true, "face_data");
	return df.session();
}

std::shared_ptr<Session> create_face_labels() {
	DeepFlow df;
	df.data_generator(df.random_uniform({ FLAGS_batch, 1, 1, 1 }, 0.8, 1.0), FLAGS_total, "", "face_labels");	
	return df.session();
}

std::shared_ptr<Session> create_generator_labels() {
	DeepFlow df;
	df.data_generator(df.random_uniform({ FLAGS_batch, 1, 1, 1 }, 0, 0.1), FLAGS_total, "", "generator_labels");
	return df.session();
}

std::shared_ptr<Session> create_loss() {
	DeepFlow df;
	auto discriminator_input = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "discriminator_input");
	auto labels_input = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "labels_input");
	auto sqerr = df.square_error(discriminator_input, labels_input);
	auto loss = df.loss(sqerr, DeepFlow::SUM);	
	df.print({ loss }, " NORM {0}\n", DeepFlow::EVERY_PASS); 
	return df.session();
}

std::shared_ptr<Session> create_generator() {
	DeepFlow df;
	
	auto mean = 0;
	auto stddev = 0.1;
	auto negative_slope = 0;
	auto alpha_param = 0.01;
	auto dropout = 0.7;
	auto exp_avg_factor = 0;
	int depth = 128;
	int z = 100;

	auto g_solver = df.adam_solver(0.0001f, 0.5f, 0.9999f);
	auto gin = df.data_generator(df.random_normal({ FLAGS_batch, z, 1, 1 }, mean, stddev, "random_input"), FLAGS_total, "", "input");
	
	auto gfc_w = df.variable(df.random_normal({ z, depth, 4, 4 }, mean, stddev), g_solver, "gfc_w");
	auto gfc = df.matmul(gin, gfc_w, "gfc");	
	auto gfc_r = df.leaky_relu(gfc, negative_slope);
	auto gfc_d = df.dropout(gfc_r, dropout);

	auto gconv1_f = df.variable(df.random_normal({ depth, depth/2, 3, 3 }, mean, stddev), g_solver, "gconv1_f");
	auto gconv1_t = df.transposed_conv2d(gfc_d, gconv1_f, 1, 1, 2, 2, 1, 1, "gconv1");	
	auto gconv1_n = df.batch_normalization(gconv1_t, DeepFlow::SPATIAL, exp_avg_factor, 1, 0, alpha_param, 1);
	auto gconv1_r = df.leaky_relu(gconv1_n, negative_slope);
	auto gconv1_d = df.dropout(gconv1_r, dropout);

	auto gconv2_f = df.variable(df.random_normal({ depth/2, depth/4, 3, 3 }, mean, stddev), g_solver, "gconv2_f");
	auto gconv2_t = df.transposed_conv2d(gconv1_d, gconv2_f, 1, 1, 2, 2, 1, 1, "gconv2");	
	auto gconv2_n = df.batch_normalization(gconv2_t, DeepFlow::SPATIAL, exp_avg_factor, 1, 0, alpha_param, 1);
	auto gconv2_r = df.leaky_relu(gconv2_n, negative_slope);
	auto gconv2_d = df.dropout(gconv2_r, dropout);

	auto gconv3_f = df.variable(df.random_normal({ depth/4, depth/8, 3, 3 }, mean, stddev), g_solver, "gconv3_f");
	auto gconv3_t = df.transposed_conv2d(gconv2_d, gconv3_f, 1, 1, 2, 2, 1, 1, "gconv3");		
	auto gconv3_n = df.batch_normalization(gconv3_t, DeepFlow::SPATIAL, exp_avg_factor, 1, 0, alpha_param, 1);
	auto gconv3_r = df.leaky_relu(gconv3_n, negative_slope);
	auto gconv3_d = df.dropout(gconv3_r, dropout);

	auto gconv4_f = df.variable(df.random_normal({ depth/8, FLAGS_channels, 3, 3 }, mean, stddev), g_solver, "gconv3_f");
	auto gconv4_t = df.transposed_conv2d(gconv3_d, gconv4_f, 1, 1, 2, 2, 1, 1, "gconv3");	
	auto gconv4_n = df.batch_normalization(gconv4_t, DeepFlow::SPATIAL, exp_avg_factor, 1, 0, alpha_param, 1);
	
	auto gout = df.tanh(gconv4_n, "gout");

	auto disp = df.display(gout, 1, DeepFlow::EVERY_PASS, DeepFlow::VALUES);

	return df.session();
}

std::shared_ptr<Session> create_discriminator() {
	DeepFlow df;
	auto mean = 0;
	auto stddev = 0.02;
	auto d_solver = df.adam_solver(0.0002f, 0.5f, 0.999f);
	auto negative_slope = 0.1;
	int depth = 32;

	auto input = df.place_holder({ FLAGS_batch , FLAGS_channels, 64, 64 }, Tensor::Float, "input");
	
	auto conv1_w = df.variable(df.random_normal({ depth, FLAGS_channels, 3, 3 }, mean, stddev), d_solver, "conv1_w");
	auto conv1 = df.conv2d(input, conv1_w, 1, 1, 2, 2, 1, 1, "conv1");	
	auto conv1_r = df.leaky_relu(conv1, negative_slope);
	auto conv1_d = df.dropout(conv1_r);

	auto conv2_w = df.variable(df.random_normal({ depth, depth, 3, 3 }, mean, stddev), d_solver, "conv2_w");
	auto conv2 = df.conv2d(conv1_d, conv2_w, 1, 1, 2, 2, 1, 1, "conv2");	
	auto conv2_r = df.leaky_relu(conv2, negative_slope);
	auto conv2_d = df.dropout(conv2_r);

	auto conv3_w = df.variable(df.random_normal({ depth*2, depth, 3, 3 }, mean, stddev), d_solver, "conv3_w");
	auto conv3 = df.conv2d(conv2_d, conv3_w, 1, 1, 2, 2, 1, 1, "conv3");	
	auto conv3_r = df.leaky_relu(conv3, negative_slope);	
	auto conv3_d = df.dropout(conv3_r);

	auto w1 = df.variable(df.random_normal({ depth*2*8*8, 500, 1, 1 }, mean, stddev), d_solver, "w1");
	auto m1 = df.matmul(conv3_d, w1, "m1");
	auto b1 = df.variable(df.step({ 1, 500, 1, 1 }, mean, stddev), d_solver, "b1");
	auto bias1 = df.bias_add(m1, b1, "bias1");
	auto relu1 = df.leaky_relu(bias1, negative_slope, "relu1");		
	auto drop1 = df.dropout(relu1);

	auto w2 = df.variable(df.random_normal({ 500, 1, 1, 1 }, mean, stddev), d_solver, "w2");
	auto m2 = df.matmul(drop1, w2, "m2");
	auto b2 = df.variable(df.step({ 1, 1, 1, 1 }, mean, stddev), d_solver, "b2");
	auto bias2 = df.bias_add(m2, b2, "bias2");

	auto dout = df.sigmoid(bias2, "dout");
	return df.session();
}

void main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();	
	
	auto execution_context = std::make_shared<ExecutionContext>();
	execution_context->debug_level = FLAGS_debug;

	auto generator = create_generator();
	generator->initialize();
	generator->set_execution_context(execution_context);
	auto discriminator = create_discriminator();	
	discriminator->initialize();
	discriminator->set_execution_context(execution_context);
	auto face_reader = create_face_reader();
	face_reader->initialize();
	face_reader->set_execution_context(execution_context);
	auto face_labels = create_face_labels();
	face_labels->initialize();
	face_labels->set_execution_context(execution_context);
	auto generator_labels = create_generator_labels();
	generator_labels->initialize();
	generator_labels->set_execution_context(execution_context);
	auto loss = create_loss();
	loss->initialize();
	loss->set_execution_context(execution_context);

	auto face_reader_data = face_reader->get_node("face_data");
	auto generator_output = generator->get_node("gout");
	auto discriminator_input = discriminator->get_placeholder("input");
	auto discriminator_output = discriminator->get_node("dout");
	auto generator_labels_output = generator_labels->get_node("generator_labels");
	auto face_labels_output = face_labels->get_node("face_labels");
	auto loss_discriminator_input = loss->get_placeholder("discriminator_input");
	auto loss_labels_input = loss->get_placeholder("labels_input");

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> select_dist(0, 2);

	for (int i = 1; i <= FLAGS_epoch && execution_context->quit != true; ++i) {
		std::cout << "Epoch: " << i << std::endl;		
		for (int k = 1; k <= 2; ++k) {			
			auto select = select_dist(gen);
			if (select == 0) { 
				// GENERATOR
				std::cout << " GENERATOR INPUT " << std::endl;
				generator->forward();
				discriminator_input->write_values(generator_output->output(0)->value());
				discriminator->forward();
				generator_labels->forward();
				loss_discriminator_input->write_values(discriminator_output->output(0)->value());
				loss_labels_input->write_values(generator_labels_output->output(0)->value());
			}
			else { 
				// FACE
				std::cout << " FACE INPUT " << std::endl;
				face_reader->forward();
				discriminator_input->write_values(face_reader_data->output(0)->value());
				discriminator->forward();
				face_labels->forward();
				loss_discriminator_input->write_values(discriminator_output->output(0)->value());
				loss_labels_input->write_values(face_labels_output->output(0)->value());
			}			
			loss->forward();
			loss->backward();
			discriminator->reset_gradients();
			discriminator_output->write_diffs(loss_discriminator_input->output(0)->diff());
			discriminator->backward();
			discriminator->apply_solvers();
		}
		std::cout << " TRAINING GENERATOR " << std::endl;
		generator->forward();
		discriminator_input->write_values(generator_output->output(0)->value());
		discriminator->forward();
		face_labels->forward();		
		loss_discriminator_input->write_values(discriminator_output->output(0)->value());
		loss_labels_input->write_values(face_labels_output->output(0)->value());
		loss->forward();
		loss->backward();
		discriminator->reset_gradients();
		discriminator_output->write_diffs(loss_discriminator_input->output(0)->diff());
		discriminator->backward();
		generator->reset_gradients();
		generator_output->write_diffs(discriminator_input->output(0)->diff());
		generator->backward();
		generator->apply_solvers();
	}

}

