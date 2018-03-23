
#include "core/deep_flow.h"
#include "core/session.h"

#include <gflags/gflags.h>

DEFINE_string(mnist, "C:/Projects/deepflow/data/mnist", "Path to MNIST data folder");
DEFINE_string(celeba128, "C:/Projects/deepflow/data/celeba128", "Path to face dataset folder");
DEFINE_string(image1, "C:/Projects/deepflow/data/image/lena-256x256.jpg", "Input test image");
DEFINE_string(image2, "C:/Projects/deepflow/data/style_transfer/sanfrancisco.jpg", "Another input test image");
DEFINE_string(image_folder, "C:/Projects/deepflow/data/face", "Path to image folder");
DEFINE_string(model, "C:/Projects/deepflow/models/VGG_ILSVRC_16_layers.caffemodel", "Caffe model to load");

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
DEFINE_bool(x2, false, "Transposed convolution gray image reconstruction");
DEFINE_bool(x3, false, "Transposed convolution color image reconstruction");
DEFINE_bool(x4, false, "Random selector double image reconstruction");
DEFINE_bool(x5, false, "Test of image_batch_reader");
DEFINE_bool(x6, false, "Test reading caffe model");
DEFINE_bool(x7, false, "Test convolution forward with bias");
DEFINE_bool(x8, false, "Test color image display");
DEFINE_bool(x9, false, "Test restructured image display");
DEFINE_bool(x10, false, "Test batch normalization");
DEFINE_bool(x11, false, "Test lifting");
DEFINE_bool(x12, false, "Test patching");
DEFINE_bool(x13, false, "Test upsample");
DEFINE_bool(x14, false, "Test Concate");
DEFINE_bool(x15, false, "Test resize");
DEFINE_bool(x16, false, "Test Gaussian weights 1");
DEFINE_bool(x17, false, "Test Gaussian weights 2");

void main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();	

	int batch_size = FLAGS_batch;
	
	DeepFlow df;
		
	if (FLAGS_i.empty()) {
		if (FLAGS_x1) {
			auto train = df.define_train_phase("Train");						
			auto solver = df.adam_solver(0.02f, 0.5f, 0.5f);
			//auto solver = df.sgd_solver(0.99f, 0.0001f);
			auto image = df.imread(FLAGS_image1, deepflow::ImageReaderParam_Type_COLOR_IF_AVAILABLE);
			auto generator = df.data_generator(df.random_uniform({ 1, 3, 256, 256 }, -1, 1), solver, "gen");
			auto euclidean = df.square_error(image, generator);
			auto loss = df.loss(euclidean, DeepFlow::AVG);
			df.display(image, 2, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1,  "input", { train });
			df.display(generator, 2, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "approximation", { train });
			df.psnr(image, generator, DeepFlow::EVERY_PASS);
		}
		else if (FLAGS_x2) {
			auto train = df.define_train_phase("Train");		
			auto solver = df.adam_solver(0.002f, 0.9f, 0.99f);
			//auto solver = df.gain_solver(0.98, 0.000001f);
			auto image = df.imread(FLAGS_image1, deepflow::ImageReaderParam_Type_GRAY_ONLY, "image");
			auto recon = df.variable(df.random_normal({ 1,1,256,256 }, 0, 0.1), solver, "recon");
			auto f1 = df.variable(df.step({ 11,1,5,5 }, 0, 1), "" , "w");
			auto conv = df.conv2d(image, f1, 2, 2, 1, 1, 1, 1);			
			auto f2 = df.variable(df.step({ 1,11,5,5 }, 0, 1), "", "w");
			auto tconv = df.transposed_conv2d(recon, f2, 2, 2, 1, 1, 1, 1);
			auto euclidean = df.square_error(conv, tconv);
			auto loss = df.loss(euclidean, DeepFlow::AVG);			
			df.display(recon, 20, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "input", { train });
			df.psnr(recon, image, DeepFlow::EVERY_PASS, "psnr", { train });
		}
		else if (FLAGS_x3) {			
			auto train = df.define_train_phase("Train");			
			auto solver = df.adam_solver(0.01f, 0.5f, 0.9f);
			auto image = df.imread(FLAGS_image1, deepflow::ImageReaderParam_Type_COLOR_IF_AVAILABLE, "image");
			auto recon = df.variable(df.random_normal({ 1,3,256,256 }, 0, 0.1), solver, "recon");			
			auto f1 = df.data_generator(df.random_uniform({ 10,3,3,3 }, -1, 1), "", "f1");
			auto f2 = df.restructure(f1, 0, 1);
			auto conv = df.conv2d(image, f1, "", 1, 1, 1, 1, 1, 1);			
			auto tconv = df.transposed_conv2d(recon, f2, 1, 1, 1, 1, 1, 1);
			auto euclidean = df.square_error(conv, tconv);
			auto loss = df.loss(euclidean, DeepFlow::SUM);
			df.display(recon, 2, DeepFlow::END_OF_EPOCH, DeepFlow::VALUES, 1, "input", { train });
			df.psnr(recon, image, DeepFlow::END_OF_EPOCH, "psnr", { train });
		}
		else if (FLAGS_x4) {
			auto train = df.define_train_phase("Train");			
			//auto solver = df.gain_solver(1.0f, 0.01f, 100, 0.0000001f, 0.05f, 0.95f);
			auto solver = df.adadelta_solver();
			//auto solver = df.adam_solver(0.1f, 0.5f, 0.9f);
			auto image = df.imread(FLAGS_image1, deepflow::ImageReaderParam_Type_COLOR_IF_AVAILABLE);
			auto generator1 = df.data_generator(df.random_uniform({ 1, 3, 256, 256 }, -0.1, 0.1), solver, "gen1");
			auto generator2 = df.data_generator(df.random_normal({ 1, 3, 256, 256 }, 0, 0.1), solver, "gen2");
			auto selector = df.random_selector(generator1, generator2, 0.5);
			auto euclidean = df.square_error(selector, image);
			auto loss = df.loss(euclidean, DeepFlow::SUM);			
			df.display(generator1, 1, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "approx1", { train });
			df.display(generator2, 1, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "approx2", { train });
		}
		else if (FLAGS_x5) {
			auto train = df.define_train_phase("Train");
			//auto imbar = df.image_batch_reader(FLAGS_image_folder, { 30, 1, 27, 18 }, true);
			auto imbar = df.image_batch_reader(FLAGS_celeba128, { 50, 3, 128, 128 }, true);
			df.display(imbar, 500, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "approx1", { train });
		}
		else if (FLAGS_x6) {
			df.load_from_caffe_model(FLAGS_model, { std::pair<std::string, std::array<int,4>>("data", {4,3,224,224}) }, FLAGS_debug > 0);			
			df.block()->remove_node_params({ "fc6_ip", "fc6_w", "fc6_b", "fc7_w", "fc7_b", "fc8_w", "fc8_b" });	
			auto train = df.define_train_phase("Train");
			auto solver = df.gain_solver(1.0f, 0.01f, 100, 0.000000001f, 0.05f, 0.95f);
			df.block()->set_solver_for_variable_params(solver, {});
			df.block()->set_phase_for_node_params( train, {});
		}
		else if (FLAGS_x7) {
			auto train = df.define_train_phase("Train");
			auto image = df.imread(FLAGS_image1, deepflow::ImageReaderParam_Type_GRAY_ONLY);
			auto b = df.variable(df.ones({ 1,1,1,1 }), "", "b", {});
			auto f = df.variable(df.ones({ 1,1,5,5 }), "", "f", {});
			auto conv1 = df.conv2d(image, f, b, -1.0, 0, 0, 1, 1, 1, 1, "conv");
			df.display(conv1, 20, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "input", { train });
			auto conv2 = df.conv2d(image, f, b, 0.0, 0, 0, 1, 1, 1, 1, "conv");
			df.display(conv2, 20, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "input", { train });
			df.logger({ conv1 }, "log.txt", "{0}\n", DeepFlow::EVERY_PASS);
			df.logger({ conv2 }, "log.txt", "{0}\n", DeepFlow::EVERY_PASS);
		}
		else if (FLAGS_x8) {
			auto train = df.define_train_phase("Train");
			auto image = df.imread(FLAGS_image1, deepflow::ImageReaderParam_Type_COLOR_IF_AVAILABLE);
			df.display(image, 10000, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "input", { train });
		}
		else if (FLAGS_x9) {
			auto train = df.define_train_phase("Train");
			auto image = df.imread(FLAGS_image1, deepflow::ImageReaderParam_Type_COLOR_IF_AVAILABLE);
			auto rotate = df.restructure(image, 2, 3);
			df.display(rotate, 10000, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1,  "rotate", { train });
		}
		else if (FLAGS_x10) {
			auto train = df.define_train_phase("Train");
			auto image = df.imread(FLAGS_image1, deepflow::ImageReaderParam_Type_COLOR_IF_AVAILABLE);
			auto scale = df.variable(df.random_normal({ 1, 3, 1, 1 }, 0, 1));
			auto bias = df.variable(df.random_normal({ 1, 3, 1, 1 }, 0, 1));
			auto norm = df.batch_normalization(image, scale, bias, DeepFlow::SPATIAL, true);
			df.display(norm, 10000, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "disp", { train });
		}
		else if (FLAGS_x11) {
			auto train = df.define_train_phase("Train");
			auto image = df.imread(FLAGS_image1, deepflow::ImageReaderParam_Type_COLOR_IF_AVAILABLE);
			auto liftdown = df.lifting(image, DeepFlow::LIFT_DOWN_FLIP);			
			df.display(liftdown, 10000, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "DOWN", { train });
			auto liftup = df.lifting(liftdown, DeepFlow::LIFT_UP_FLIP);			
			df.display(liftup, 10000, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "UP", { train });
		}
		else if (FLAGS_x12) {
			
			/*
			auto train = df.define_train_phase("Train");
			//auto image = df.image_reader(FLAGS_image1, deepflow::ImageReaderParam_Type_COLOR_IF_AVAILABLE);			
			auto image = df.image_reader(FLAGS_image1, deepflow::ImageReaderParam_Type_GRAY_ONLY);
			auto patchdown = df.patching(image, DeepFlow::PATCHING_DOWN, 2, 2);
			auto patchup = df.patching(patchdown, DeepFlow::PATCHING_UP, 2, 2);
			df.display(patchup, 10000, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "disp", { train });
			*/

			
			auto train = df.define_train_phase("Train");
			auto images = df.image_batch_reader(FLAGS_image_folder, { 6, 1, 27, 18 }, true);
			auto patchdown = df.patching(images, DeepFlow::PATCHING_DOWNCHANNELS, 3, 2);
			auto patchup = df.patching(patchdown, DeepFlow::PATCHING_UPCHANNELS, 3, 2);
			df.display(patchup, 10000, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "disp1", { train });
			
		}
		else if (FLAGS_x13) {
			auto train = df.define_train_phase("Train");
			auto image = df.imread(FLAGS_image1, deepflow::ImageReaderParam_Type_COLOR_IF_AVAILABLE);
			auto up = df.upsample(image);
			df.display(up, 10000, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "disp", { train });
		}
		else if (FLAGS_x14) {
			auto train = df.define_train_phase("Train");
			auto images1 = df.image_batch_reader(FLAGS_image_folder, { 20, 1, 27, 18 }, true);
			auto images2 = df.image_batch_reader(FLAGS_image_folder, { 20, 1, 27, 18 }, true);			
			auto concate = df.concate(images1, images2);
			df.display(concate, 5000, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "disp", { train });
		}
		else if (FLAGS_x15) {
			auto train = df.define_train_phase("Train");
			auto images = df.image_batch_reader(FLAGS_image_folder, { 20, 1, 27, 18 }, true);			
			auto resized = df.resize(images, 2.0, 2.0);
			df.display(resized, 5000, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "disp", { train });
		}
		else if (FLAGS_x16) {
			auto train = df.define_train_phase("Train");
			auto gw = df.gaussian_kernel(64, 10);
			df.display(gw, 5000, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "disp", { train });
		}
		else if (FLAGS_x17) {
			auto train = df.define_train_phase("Train");
			auto imba = df.image_batch_reader(FLAGS_celeba128, { 50, 3, 128, 128 }, true);
			//auto gw = df.gaussian_kernel(63, 0.1);
			auto gw = df.gaussian_kernel(63, 10);
			auto conv = df.conv2d(imba, gw, 31, 31, 1, 1, 1, 1);
			df.display(conv, 5000, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "disp", { train });
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


