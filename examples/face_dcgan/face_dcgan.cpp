
#include "core/deep_flow.h"
#include "core/session.h"

#include <random>
#include <gflags/gflags.h>

DEFINE_string(faces, "D:/Projects/deepflow/data/face64", "Path to face 64x64 dataset folder");
DEFINE_string(i, "", "Trained network model to load");
DEFINE_string(o, "", "Trained network model to save");
DEFINE_bool(text, false, "Save model as text");
DEFINE_bool(includeweights, false, "Also save weights in text mode");
DEFINE_bool(includeinits, false, "Also save initial values");
DEFINE_int32(batch, 80, "Batch size");
DEFINE_string(run,"", "Phase to execute graph");
DEFINE_bool(printiter, false, "Print iteration message");
DEFINE_bool(printepoch, true, "Print epoch message");
DEFINE_int32(debug, 0, "Level of debug");
DEFINE_int32(epoch, 1000000, "Maximum epochs");
DEFINE_int32(iter, -1, "Maximum iterations");
DEFINE_bool(cpp, false, "Print C++ code");

std::shared_ptr<Session> create_face_reader() {
	DeepFlow df;
	df.image_batch_reader(FLAGS_faces, { FLAGS_batch, 1, 64, 64 }, true, "face_data");
	return df.session();
}

std::shared_ptr<Session> create_face_labels() {
	DeepFlow df;
	df.data_generator(df.random_uniform({ FLAGS_batch, 1, 1, 1 }, 0.7, 1.0), 240, "", "face_labels");	
	return df.session();
}

std::shared_ptr<Session> create_generator_labels() {
	DeepFlow df;
	df.data_generator(df.random_uniform({ FLAGS_batch, 1, 1, 1 }, 0, 0.3), 240, "", "generator_labels");
	return df.session();
}

std::shared_ptr<Session> create_loss() {
	DeepFlow df;
	auto discriminator_input = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "discriminator_input");
	auto labels_input = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "labels_input");
	df.euclidean_loss(discriminator_input, labels_input);
	auto sub = df.subtract(discriminator_input, labels_input);
	auto reduce = df.reduce_norm1(sub, 0);
	df.print({ reduce }, " NORM {0}\n", DeepFlow::EVERY_PASS); 
	return df.session();
}

std::shared_ptr<Session> create_generator() {
	DeepFlow df;
	
	auto mean = 0;
	auto stddev = 0.02;

	auto g_solver = df.adadelta_solver(0.01f, 0.4f);// gain_solver();// adam_solver(0.001f, 0.5f, 0.75f);
	auto gin = df.data_generator(df.random_normal({ FLAGS_batch, 100, 1, 1 }, mean, stddev, "random_input"), 240, "", "input");
	
	auto gfc_w = df.variable(df.random_normal({ 100, 256, 4, 4 }, mean, stddev), g_solver, "gfc_w");
	auto gfc = df.matmul(gin, gfc_w, "gfc");
	auto gfc_d = df.dropout(gfc, 0.2f);
	auto gfc_r = df.leaky_relu(gfc_d);

	auto gconv1_f = df.variable(df.random_normal({ 256, 128, 3, 3 }, mean, stddev), g_solver, "gconv1_f");
	auto gconv1_t = df.transposed_conv2d(gfc_r, gconv1_f, 1, 1, 2, 2, 1, 1, "gconv1");
	auto gconv1_n = df.batch_normalization(gconv1_t, DeepFlow::PER_ACTIVATION);
	auto gconv1_r = df.leaky_relu(gconv1_n);
	
	auto gconv2_f = df.variable(df.random_normal({ 128, 64, 3, 3 }, mean, stddev), g_solver, "gconv2_f");
	auto gconv2_t = df.transposed_conv2d(gconv1_r, gconv2_f, 1, 1, 2, 2, 1, 1, "gconv2");
	auto gconv2_n = df.batch_normalization(gconv2_t, DeepFlow::PER_ACTIVATION);
	auto gconv2_r = df.leaky_relu(gconv2_n);

	auto gconv3_f = df.variable(df.random_normal({ 64, 32, 3, 3 }, mean, stddev), g_solver, "gconv3_f");
	auto gconv3_t = df.transposed_conv2d(gconv2_r, gconv3_f, 1, 1, 2, 2, 1, 1, "gconv3");	
	auto gconv3_n = df.batch_normalization(gconv3_t, DeepFlow::PER_ACTIVATION);
	auto gconv3_r = df.leaky_relu(gconv3_n);

	auto gconv4_f = df.variable(df.random_normal({ 32, 1, 3, 3 }, mean, stddev), g_solver, "gconv3_f");
	auto gconv4_t = df.transposed_conv2d(gconv3_r, gconv4_f, 1, 1, 2, 2, 1, 1, "gconv3");
	auto gconv4_n = df.batch_normalization(gconv4_t, DeepFlow::PER_ACTIVATION);

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
	auto input = df.place_holder({ FLAGS_batch , 1, 64, 64 }, Tensor::Float, "input");	
	
	auto conv1_w = df.variable(df.random_uniform({ 16, 1, 3, 3 }, -0.100000, 0.100000), d_solver, "conv1_w");
	auto conv1 = df.conv2d(input, conv1_w, 1, 1, 2, 2, 1, 1, "conv1");
	//auto conv1_n = df.batch_normalization(conv1, DeepFlow::PER_ACTIVATION);
	auto conv1_r = df.leaky_relu(conv1, negative_slope);
	
	auto conv2_w = df.variable(df.random_uniform({ 32, 16, 3, 3 }, -0.100000, 0.100000), d_solver, "conv2_w");
	auto conv2 = df.conv2d(conv1_r, conv2_w, 1, 1, 2, 2, 1, 1, "conv2");
	//auto conv2_n = df.batch_normalization(conv2, DeepFlow::PER_ACTIVATION);
	auto conv2_r = df.leaky_relu(conv2, negative_slope);

	auto conv3_w = df.variable(df.random_uniform({ 64, 32, 3, 3 }, -0.100000, 0.100000), d_solver, "conv3_w");
	auto conv3 = df.conv2d(conv2_r, conv3_w, 1, 1, 2, 2, 1, 1, "conv3");
	//auto conv3_n = df.batch_normalization(conv3, DeepFlow::PER_ACTIVATION);
	auto conv3_r = df.leaky_relu(conv3, negative_slope);

	auto w1 = df.variable(df.random_uniform({ 4096, 500, 1, 1 }, -0.100000, 0.100000), d_solver, "w1");
	auto m1 = df.matmul(conv3_r, w1, "m1");
	auto b1 = df.variable(df.step({ 1, 500, 1, 1 }, -1.000000, 1.000000), d_solver, "b1");
	auto bias1 = df.bias_add(m1, b1, "bias1");
	auto relu1 = df.leaky_relu(bias1, negative_slope, "relu1");
	
	auto w2 = df.variable(df.random_uniform({ 500, 1, 1, 1 }, -0.100000, 0.100000), d_solver, "w2");
	auto m2 = df.matmul(relu1, w2, "m2");
	auto b2 = df.variable(df.step({ 1, 1, 1, 1 }, -1.000000, 1.000000), d_solver, "b2");
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
			discriminator->reset_gradients();
			loss->reset_gradients();
			auto select = select_dist(gen);
			if (select == 0) { 
				// GENERATOR
				std::cout << " GENERATOR INPUT " << std::endl;
				generator->forward();
				discriminator_input->feed_forward(generator_output, 0);
				discriminator->forward();
				generator_labels->forward();
				loss_discriminator_input->feed_forward(discriminator_output, 0);
				loss_labels_input->feed_forward(generator_labels_output, 0);
			}
			else { 
				// FACE
				std::cout << " FACE INPUT " << std::endl;
				face_reader_data->forward();
				discriminator_input->feed_forward(face_reader_data, 0);
				discriminator->forward();
				face_labels->forward();
				loss_discriminator_input->feed_forward(discriminator_output, 0);
				loss_labels_input->feed_forward(face_labels_output, 0);				
			}			
			loss->forward();
			loss->backward();
			discriminator_output->feed_backward(loss_discriminator_input, 0);
			discriminator->backward();
			discriminator->apply_solvers();
		}
		std::cout << " TRAINING GENERATOR " << std::endl;
		generator->reset_gradients();
		discriminator->reset_gradients();
		loss->reset_gradients();
		generator->forward();
		discriminator_input->feed_forward(generator_output, 0);
		discriminator->forward();
		face_labels->forward();		
		loss_discriminator_input->feed_forward(discriminator_output, 0);
		loss_labels_input->feed_forward(face_labels_output, 0);
		loss->forward();
		loss->backward();
		discriminator_output->feed_backward(loss_discriminator_input, 0);
		discriminator->backward();
		generator_output->feed_backward(discriminator_input, 0);
		generator->backward();
		generator->apply_solvers();
	}

}

