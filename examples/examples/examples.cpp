
#include "core/deep_flow.h"
#include "core/session.h"

#include <gflags/gflags.h>

DEFINE_string(mnist, "./data/mnist", "Path to MNIST data folder");
DEFINE_string(image1, "D:/Projects/deepflow/data/image/lena-256x256.jpg", "Input test image");
DEFINE_string(image2, "D:/Projects/deepflow/data/style_transfer/sanfrancisco.jpg", "Another input test image");
DEFINE_string(model, "D:/Projects/deepflow/models/VGG_ILSVRC_16_layers.caffemodel", "Caffe model to load");

DEFINE_string(i, "", "Trained network model to load");
DEFINE_string(o, "", "Trained network model to save");
DEFINE_bool(text, false, "Save model as text");
DEFINE_bool(includeweights, false, "Also save weights in text mode");
DEFINE_bool(includeinits, false, "Also save initial values");
DEFINE_int32(batch, 100, "Batch size");
DEFINE_string(run,"", "Phase to execute graph");
DEFINE_bool(printiter, false, "Print iteration message");
DEFINE_bool(printepoch, true, "Print epoch message");
DEFINE_int32(debug, 0, "Level of debug");
DEFINE_int32(epoch, 1000, "Maximum epochs");
DEFINE_int32(iter, -1, "Maximum iterations");
DEFINE_bool(cpp, false, "Print C++ code");
DEFINE_bool(x1, false, "Eucliean image reconstruction");
DEFINE_bool(x2, false, "Transposed convolution image reconstruction");
DEFINE_bool(x3, false, "Random selector double image reconstruction");
DEFINE_bool(x4, false, "Test of image_batch_reader");
DEFINE_bool(x5, false, "Test reading caffe model");
DEFINE_bool(x6, false, "Test convolution forward with bias");
DEFINE_bool(x7, false, "Test color image display");

void main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();	

	int batch_size = FLAGS_batch;
	
	DeepFlow df;
		
	if (FLAGS_i.empty()) {
		if (FLAGS_x1) {
			auto train = df.define_train_phase("Train");			
			auto solver = df.gain_solver(1.0f, 0.01f, 100, 0.000000001f, 0.05f, 0.95f);
			//auto solver = df.sgd_solver(1.0f, 0.01f);
			//auto solver = df.adam_solver();
			//auto solver = df.adadelta_solver();
			auto image = df.image_reader(FLAGS_image1, deepflow::ImageReaderParam_Type_COLOR_IF_AVAILABLE);
			auto generator = df.data_generator(df.random_uniform({ 1, 3, 256, 256 }, -1, 1), 1, solver, "gen");
			df.euclidean_loss(generator, image);
			df.display(image, 2, deepflow::DisplayParam_DisplayType_VALUES, "input", { train });
			df.display(generator, 2, deepflow::DisplayParam_DisplayType_VALUES, "approximation", { train });
			df.psnr(image, generator, Psnr::EVERY_PASS);
		}
		else if (FLAGS_x2) {
			auto train = df.define_train_phase("Train");			
			auto solver1 = df.gain_solver(0.999f, 0.0001f, 100, 0.000000001f, 0.05f, 0.95f);
			auto solver2 = df.sgd_solver(1.0f, 0.0000000001f);
			//auto solver = df.adam_solver();
			//auto solver = df.adadelta_solver();		
			auto image = df.image_reader(FLAGS_image1, deepflow::ImageReaderParam_Type_GRAY_ONLY);
			auto stat = df.variable(df.random_uniform({ 1,1,252,252 }, 0.8, 1), solver1);
			auto f = df.variable(df.random_uniform({ 1,1,5,5 }, 0.9, 1.1), solver2);
			auto tconv = df.transposed_conv2d(stat, f, { 1,1,256,256 }, 0, 0, 1, 1, 1, 1);
			df.euclidean_loss(tconv, image);
			df.display(tconv, 20, deepflow::DisplayParam_DisplayType_VALUES, "input", { train });
			df.psnr(tconv, image, Psnr::EVERY_PASS, "psnr", { train });
		}
		else if (FLAGS_x3) {
			auto train = df.define_train_phase("Train");			
			auto solver = df.gain_solver(1.0f, 0.01f, 100, 0.000000001f, 0.05f, 0.95f);
			//auto solver = df.sgd_solver(1.0f, 0.01f);
			//auto solver = df.adam_solver();
			//auto solver = df.adadelta_solver();
			auto image = df.image_reader(FLAGS_image1, deepflow::ImageReaderParam_Type_GRAY_ONLY);
			auto generator1 = df.data_generator(df.random_uniform({ 1, 1, 256, 256 }, -0.1, 0.1), 1, solver, "gen1");
			auto generator2 = df.data_generator(df.random_normal({ 1, 1, 256, 256 }, 0, 0.1), 1, solver, "gen2");
			auto selector = df.random_selector(generator1, generator2, 0.5);
			df.euclidean_loss(selector, image);
			df.display(generator1, 2, deepflow::DisplayParam_DisplayType_VALUES, "approx1", { train });
			df.display(generator2, 2, deepflow::DisplayParam_DisplayType_VALUES, "approx2", { train });
		}
		else if (FLAGS_x4) {
			auto train = df.define_train_phase("Train");
			auto imbar = df.image_batch_reader("./data/face", { 1, 1, 27, 18 });
			df.display(imbar, 1000, deepflow::DisplayParam_DisplayType_VALUES, "approx1", { train });
		}
		else if (FLAGS_x5) {
			df.load_from_caffe_model(FLAGS_model, { std::pair<std::string, std::array<int,4>>("data", {4,3,224,224}) }, FLAGS_debug > 0);			
			df.block()->remove_node_params({ "fc6_ip", "fc6_w", "fc6_b", "fc7_w", "fc7_b", "fc8_w", "fc8_b" });	
			auto train = df.define_train_phase("Train");
			auto solver = df.gain_solver(1.0f, 0.01f, 100, 0.000000001f, 0.05f, 0.95f);
			df.block()->set_solver_for_variable_params(solver, {});
			df.block()->set_phase_for_node_params( train, {});
		}
		else if (FLAGS_x6) {
			auto train = df.define_train_phase("Train");
			auto image = df.image_reader(FLAGS_image1, deepflow::ImageReaderParam_Type_GRAY_ONLY);
			auto b = df.variable(df.zeros({ 1,1,1,1 }), "", "b", {});
			auto f = df.variable(df.ones({ 1,1,5,5 }), "", "f", {});
			auto conv = df.conv2d(image, f, b, "conv");			
			df.display(conv, 20, deepflow::DisplayParam_DisplayType_VALUES, "input", { train });
		}
		else if (FLAGS_x7) {
			auto train = df.define_train_phase("Train");
			auto image = df.image_reader(FLAGS_image2, deepflow::ImageReaderParam_Type_COLOR_IF_AVAILABLE);
			df.display(image, 10000, deepflow::DisplayParam_DisplayType_VALUES, "input", { train });
		}
	}
	else {
		df.block()->load_from_binary(FLAGS_i);
	}

	if (FLAGS_debug > 0) {
		df.block()->print_node_params();
		df.block()->print_phase_params();
	}

	auto session = df.session();	

	if (FLAGS_cpp) {
		session->initialize();
		std::cout << session->to_cpp() << std::endl;
		if (FLAGS_debug == 1) {
			session->printMemory();
		}
	}

	if (!FLAGS_run.empty()) {
		session->initialize();
		session->run(FLAGS_run, FLAGS_epoch, FLAGS_iter, FLAGS_printiter, FLAGS_printepoch, FLAGS_debug);
	}
	
	if (!FLAGS_o.empty())
	{
		if (FLAGS_text)
			df.block()->save_as_text(FLAGS_o);
		else
			df.block()->save_as_binary(FLAGS_o);
	}
	
}


