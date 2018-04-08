
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
DEFINE_int32(debug, 0, "Level of debug");
DEFINE_int32(iter, 1000, "Maximum iterations");
DEFINE_bool(cpp, false, "Print C++ code");
DEFINE_bool(x1, false, "Eucliean image reconstruction");
DEFINE_bool(x2, false, "Transposed convolution gray image reconstruction");
DEFINE_bool(x3, false, "Transposed convolution color image reconstruction");
DEFINE_bool(x4, false, "Random selector double image reconstruction");
DEFINE_bool(x5, false, "Test of image_batch_reader");
DEFINE_bool(x6, false, "Test reading caffe model");
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
DEFINE_bool(x18, false, "Test Patch sampling");

#include <chrono>

void main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();	

	int batch_size = FLAGS_batch;
	
	DeepFlow df;
		
	if (FLAGS_i.empty()) {
		if (FLAGS_x1) {									
			auto solver = df.adam_solver(0.02f, 0.5f, 0.5f);
			//auto solver = df.sgd_solver(0.99f, 0.0001f);
			auto image = df.imread(FLAGS_image1, DeepFlow::IMREAD_COLOR);
			auto generator = df.data_generator(df.random_uniform({ 1, 3, 256, 256 }, -1, 1), solver, "gen");
			auto euclidean = df.square_error(image, generator);
			auto loss = df.loss(euclidean, DeepFlow::AVG);
			df.display(image, 2, DeepFlow::VALUES, 1,  "input");
			df.display(generator, 2, DeepFlow::VALUES, 1, "approximation");
			df.psnr(image, generator, "psnr");
		}
		else if (FLAGS_x2) {			
			auto solver = df.adam_solver(0.002f, 0.9f, 0.99f);
			//auto solver = df.gain_solver(0.98, 0.000001f);
			auto image = df.imread(FLAGS_image1, DeepFlow::IMREAD_GRAY, "image");
			auto recon = df.variable(df.random_normal({ 1,1,256,256 }, 0, 0.1), solver, "recon");
			auto f1 = df.variable(df.step({ 11,1,5,5 }, 0, 1), "" , "w");
			auto conv = df.conv2d(image, f1, 2, 2, 1, 1, 1, 1);
			auto f2 = df.variable(df.step({ 1,11,5,5 }, 0, 1), "", "w");
			auto tconv = df.transposed_conv2d(recon, f2, 2, 2, 1, 1, 1, 1);
			auto euclidean = df.square_error(conv, tconv);
			auto loss = df.loss(euclidean, DeepFlow::AVG);			
			df.display(recon, 20, DeepFlow::VALUES, 1, "input");
			df.psnr(recon, image, "psnr");
		}
		else if (FLAGS_x3) {			
			auto solver = df.adam_solver(0.01f, 0.5f, 0.9f);
			auto image = df.imread(FLAGS_image1, DeepFlow::IMREAD_COLOR, "image");
			auto recon = df.variable(df.random_normal({ 1,3,256,256 }, 0, 0.1), solver, "recon");			
			auto f1 = df.data_generator(df.random_uniform({ 10,3,3,3 }, -1, 1), "", "f1");
			auto f2 = df.restructure(f1, 0, 1);
			auto conv = df.conv2d(image, f1, 1, 1, 1, 1, 1);			
			auto tconv = df.transposed_conv2d(recon, f2, 1, 1, 1, 1, 1, 1);
			auto euclidean = df.square_error(conv, tconv);
			auto loss = df.loss(euclidean, DeepFlow::AVG);
			df.display(recon, 20, DeepFlow::VALUES, 1, "input");
			df.psnr(recon, image, "psnr");
		}
		else if (FLAGS_x4) {				
			//auto solver = df.gain_solver(1.0f, 0.01f, 100, 0.0000001f, 0.05f, 0.95f);
			//auto solver = df.adadelta_solver();
			auto solver = df.adam_solver(0.1f, 0.5f, 0.9f);
			auto image = df.imread(FLAGS_image1, DeepFlow::IMREAD_COLOR);
			auto generator1 = df.data_generator(df.random_uniform({ 1, 3, 256, 256 }, -0.1, 0.1), solver, "gen1");
			auto generator2 = df.data_generator(df.random_normal({ 1, 3, 256, 256 }, 0, 0.1), solver, "gen2");
			auto selector = df.random_selector(generator1, generator2, 0.5);
			auto euclidean = df.square_error(selector, image);
			auto loss = df.loss(euclidean, DeepFlow::SUM);			
			df.display(generator1, 1, DeepFlow::VALUES, 1, "approx1");
			df.display(generator2, 1, DeepFlow::VALUES, 1, "approx2");
		}
		else if (FLAGS_x5) {			
			//auto imbar = df.image_batch_reader(FLAGS_image_folder, { 30, 1, 27, 18 }, true);
			auto imbar = df.image_batch_reader(FLAGS_celeba128, { 50, 3, 128, 128 }, true);
			df.display(imbar, 500, DeepFlow::VALUES, 1, "approx1");
		}
		else if (FLAGS_x6) {
			df.load_from_caffe_model(FLAGS_model, { std::pair<std::string, std::array<int,4>>("data", {4,3,224,224}) }, FLAGS_debug > 0);			
			df.block()->remove_node_params({ "fc6_ip", "fc6_w", "fc6_b", "fc7_w", "fc7_b", "fc8_w", "fc8_b" });				
			auto solver = df.gain_solver(1.0f, 0.01f, 100, 0.000000001f, 0.05f, 0.95f);
			df.block()->set_solver_for_variable_params(solver, {});
		}
		else if (FLAGS_x8) {			
			auto image = df.imread(FLAGS_image1, DeepFlow::IMREAD_COLOR);
			df.display(image, 10000, DeepFlow::VALUES, 1, "input");
		}
		else if (FLAGS_x9) {			
			auto image = df.imread(FLAGS_image1, DeepFlow::IMREAD_COLOR);
			auto rotate = df.restructure(image, 2, 3);
			df.display(rotate, 10000, DeepFlow::VALUES, 1,  "rotate");
		}
		else if (FLAGS_x10) {			
			auto image = df.imread(FLAGS_image1, DeepFlow::IMREAD_COLOR);
			auto scale = df.variable(df.random_normal({ 1, 3, 1, 1 }, 0, 1));
			auto bias = df.variable(df.random_normal({ 1, 3, 1, 1 }, 0, 1));
			auto norm = df.batch_normalization(image, scale, bias, DeepFlow::SPATIAL, true);
			df.display(norm, 10000, DeepFlow::VALUES, 1, "disp");
		}
		else if (FLAGS_x11) {			
			auto image = df.imread(FLAGS_image1, DeepFlow::IMREAD_COLOR);
			auto liftdown = df.lifting(image, DeepFlow::LIFT_DOWN_FLIP);			
			df.display(liftdown, 10000, DeepFlow::VALUES, 1, "DOWN");
			auto liftup = df.lifting(liftdown, DeepFlow::LIFT_UP_FLIP);			
			df.display(liftup, 10000, DeepFlow::VALUES, 1, "UP");
		}
		else if (FLAGS_x12) {
			
			/*			
			//auto image = df.image_reader(FLAGS_image1, deepflow::ImageReaderParam_Type_COLOR_IF_AVAILABLE);			
			auto image = df.image_reader(FLAGS_image1, deepflow::ImageReaderParam_Type_GRAY_ONLY);
			auto patchdown = df.patching(image, DeepFlow::PATCHING_DOWN, 2, 2);
			auto patchup = df.patching(patchdown, DeepFlow::PATCHING_UP, 2, 2);
			df.display(patchup, 10000, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "disp");
			*/
			
			auto images = df.image_batch_reader(FLAGS_image_folder, { 6, 1, 27, 18 }, true);
			auto patchdown = df.patching(images, DeepFlow::PATCHING_DOWNCHANNELS, 3, 2);
			auto patchup = df.patching(patchdown, DeepFlow::PATCHING_UPCHANNELS, 3, 2);
			df.display(patchup, 10000, DeepFlow::VALUES, 1, "disp1");
			
		}
		else if (FLAGS_x13) {			
			auto image = df.imread(FLAGS_image1, DeepFlow::IMREAD_COLOR);
			auto up = df.upsample(image);
			df.display(up, 10000, DeepFlow::VALUES, 1, "disp");
		}
		else if (FLAGS_x14) {			
			auto images1 = df.image_batch_reader(FLAGS_image_folder, { 20, 1, 27, 18 }, true);
			auto images2 = df.image_batch_reader(FLAGS_image_folder, { 20, 1, 27, 18 }, true);
			auto images3 = df.image_batch_reader(FLAGS_image_folder, { 20, 1, 27, 18 }, true);
			auto concate = df.concate(images1, images2);
			concate = df.concate(concate, images3);
			df.display(concate, 5000, DeepFlow::VALUES, 1, "disp");
		}
		else if (FLAGS_x15) {			
			auto images = df.image_batch_reader(FLAGS_image_folder, { 20, 1, 27, 18 }, true);			
			auto resized = df.resize(images, 2.0, 2.0);
			df.display(resized, 5000, DeepFlow::VALUES, 1, "disp");
		}
		else if (FLAGS_x16) {			
			auto gw = df.gaussian_kernel(64, 10);
			df.display(gw, 5000, DeepFlow::VALUES, 1, "disp");
		}
		else if (FLAGS_x17) {			
			auto imba = df.image_batch_reader(FLAGS_celeba128, { 50, 3, 128, 128 }, true);
			auto gb = df.gaussian_blur(imba, 63, 10, "gaussian_blur");
			df.display(gb, 5000, DeepFlow::VALUES, 1, "disp");
		}
		else if (FLAGS_x18) { 			
			auto input = df.image_batch_reader(FLAGS_celeba128, { 40, 3, 128, 128 }, true);
			auto patches = df.patch_sampling(input, 64, 32);
			df.display(patches, 5000, DeepFlow::VALUES, 1, "disp");
		}
	}
	else {
		df.block()->load_from_binary(FLAGS_i);
	}

	if (FLAGS_debug > 0) {
		df.block()->print_node_params();		
	}

	auto session = df.session();	

	if (FLAGS_cpp) {
		session->initialize();
		std::cout << session->to_cpp() << std::endl;
	}

	session->initialize();		
	auto psnr_node = session->get_node<Psnr>("psnr");
	for (int iter = 1; iter <= FLAGS_iter; ++iter) {		
		std::cout << "Iteration " << iter << " -->" << std::endl;
		auto epoch_start = std::chrono::high_resolution_clock::now();
		session->forward(session->end_nodes());
		if (psnr_node) {
			std::cout << "PSNR: " << psnr_node->psnr() << std::endl;
		}
		session->backward(session->end_nodes());
		session->apply_solvers();
		auto epoch_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_epoch = epoch_end - epoch_start;
		std::cout << "<-- Iteration " << iter << " Elapsed time: " << elapsed_epoch.count() << " seconds" << std::endl;
		if (session->check_quit() == true)
			break;
	}
	
	if (!FLAGS_o.empty())
	{
		if (FLAGS_text)
			df.block()->save_as_text(FLAGS_o);
		else
			df.block()->save_as_binary(FLAGS_o);
	}
	
}


