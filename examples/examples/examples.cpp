
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
DEFINE_bool(x19, false, "Test Spatial Transformer 1");
DEFINE_bool(x20, false, "Test Spatial Transformer 2");

#include <chrono>

void main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();	

	int batch_size = FLAGS_batch;
	
	DeepFlow df;
		
	if (FLAGS_i.empty()) {
		if (FLAGS_x1) {									
			//auto solver = df.adam_solver(AdamSolverOp().lr(0.02f).beta1(0.5f).beta2(0.5f));
			//auto solver = df.adadelta_solver( AdaDeltaSolverOp());
			//auto solver = df.rmsprop_solver();
			auto solver = df.sgd_solver();
			auto image = df.imread(FLAGS_image1, ImreadOp().color());
			auto generator = df.data_generator(df.random_uniform({ 1, 3, 256, 256 }, -1, 1), DataGeneratorOp().solver(solver));
			auto euclidean = df.square_error(image, generator);
			auto loss = df.loss(euclidean);
			df.display(image, DisplayOp("input").delay(2));
			df.display(generator, DisplayOp("approx"));
			df.psnr(image, generator);
		}
		else if (FLAGS_x2) {			
			auto solver = df.adam_solver(AdamSolverOp().lr(0.002f).beta1(0.9f).beta2(0.9f));						
			auto image = df.imread(FLAGS_image1, ImreadOp().gray());
			auto recon = df.variable(df.random_normal({ 1,1,256,256 }, 0, 0.1), solver, VariableOp("recon"));
			auto f1 = df.variable(df.step({ 11,1,5,5 }, 0, 1), "" , VariableOp("w"));
			auto conv = df.conv2d(image, f1, ConvolutionOp().pad(2));
			auto f2 = df.variable(df.step({ 1,11,5,5 }, 0, 1), "", VariableOp("w"));
			auto tconv = df.transposed_conv2d(recon, f2, ConvolutionOp().pad(2));
			auto euclidean = df.square_error(conv, tconv);
			auto loss = df.loss(euclidean);			
			df.display(recon, DisplayOp("input").delay(20));
			df.psnr(recon, image);
		}
		else if (FLAGS_x3) {			
			auto solver = df.adam_solver(AdamSolverOp().lr(0.1f).beta1(0.5f).beta2(0.9f));
			auto image = df.imread(FLAGS_image1, ImreadOp().color());
			auto recon = df.variable(df.random_normal({ 1,3,256,256 }, 0, 0.1), solver, VariableOp("recon"));			
			auto f1 = df.data_generator(df.random_uniform({ 10,3,3,3 }, -1, 1));
			auto f2 = df.restructure(f1, 0, 1);
			auto conv = df.conv2d(image, f1, ConvolutionOp().pad(1).dilation(1).stride(1));
			auto tconv = df.transposed_conv2d(recon, f2, ConvolutionOp().pad(1).dilation(1).stride(1));
			auto euclidean = df.square_error(conv, tconv);
			auto loss = df.loss(euclidean);
			df.display(recon, DisplayOp("input").delay(20));
			df.psnr(recon, image);
		}
		else if (FLAGS_x4) {				
			auto solver = df.adam_solver( AdamSolverOp().lr(0.1f).beta1(0.5f).beta2(0.9f));
			auto image = df.imread(FLAGS_image1, ImreadOp().color());
			auto generator1 = df.data_generator(df.random_uniform({ 1, 3, 256, 256 }, -0.1, 0.1), DataGeneratorOp().solver(solver).name("gen1"));
			auto generator2 = df.data_generator(df.random_normal({ 1, 3, 256, 256 }, 0, 0.1), DataGeneratorOp().solver(solver).name("gen2"));
			auto selector = df.random_selector(generator1, generator2);
			auto euclidean = df.square_error(selector, image);
			auto loss = df.loss(euclidean, LossOp().sum());			
			df.display(generator1, DisplayOp("approx1"));
			df.display(generator2, DisplayOp("approx2"));
		}
		else if (FLAGS_x5) {			
			auto imbar = df.imbatch(FLAGS_celeba128, { 50, 3, 128, 128 });
			df.display(imbar, DisplayOp("approx1").delay(500));
		}
		else if (FLAGS_x6) {
			df.load_from_caffe_model(FLAGS_model, { std::pair<std::string, std::array<int,4>>("data", {4,3,224,224}) }, FLAGS_debug > 0);			
			df.block()->remove_node_params({ "fc6_ip", "fc6_w", "fc6_b", "fc7_w", "fc7_b", "fc8_w", "fc8_b" });				
			auto solver = df.adam_solver();
			df.block()->set_solver_for_variable_params(solver, {});
		}
		else if (FLAGS_x8) {			
			auto image = df.imread(FLAGS_image1, ImreadOp().color());
			df.display(image, DisplayOp("image").delay(10000));
		}
		else if (FLAGS_x9) {			
			auto image = df.imread(FLAGS_image1, ImreadOp().color());
			auto rotate = df.restructure(image, 2, 3);
			df.display(rotate, DisplayOp("rotated").delay(10000));
		}
		else if (FLAGS_x10) {			
			auto image = df.imread(FLAGS_image1, ImreadOp().color());
			auto scale = df.variable(df.random_normal({ 1, 3, 1, 1 }, 0, 1));
			auto bias = df.variable(df.random_normal({ 1, 3, 1, 1 }, 0, 1));
			auto norm = df.batch_normalization(image, scale, bias);
			df.display(norm, DisplayOp().delay(10000));
		}
		else if (FLAGS_x11) {			
			auto image = df.imread(FLAGS_image1, ImreadOp().color());
			auto liftdown = df.lifting(image, LiftingOp().with_flip().down());			
			df.display(liftdown, DisplayOp("DOWN").delay(10000));
			auto liftup = df.lifting(liftdown, LiftingOp().with_flip().up());			
			df.display(liftup, DisplayOp("UP").delay(10000));
		}
		else if (FLAGS_x12) {
						
			auto images = df.imbatch(FLAGS_image_folder, { 6, 1, 27, 18 });
			auto patchdown = df.patching(images, PatchingOp().channels().down().h(2).v(3));
			auto patchup = df.patching(patchdown, PatchingOp().channels().up().h(2).v(3));
			df.display(patchup, DisplayOp("disp1").delay(10000));
			
		}
		else if (FLAGS_x13) {			
			auto image = df.imread(FLAGS_image1, ImreadOp().color());
			auto up = df.resize(image, 2.0, 2.0);
			df.display(up, DisplayOp().delay(10000));
		}
		else if (FLAGS_x14) {			
			auto images1 = df.imbatch(FLAGS_image_folder, { 20, 1, 27, 18 });
			auto images2 = df.imbatch(FLAGS_image_folder, { 20, 1, 27, 18 });
			auto images3 = df.imbatch(FLAGS_image_folder, { 20, 1, 27, 18 });
			auto concate = df.concate(images1, images2);
			concate = df.concate(concate, images3);
			df.display(concate, DisplayOp().delay(5000));
		}
		else if (FLAGS_x15) {			
			auto images = df.imbatch(FLAGS_image_folder, { 20, 1, 27, 18 });			
			auto resized = df.resize(images, 2.0, 2.0);
			df.display(resized, DisplayOp().delay(5000));
		}
		else if (FLAGS_x16) {			
			auto gw = df.gaussian_kernel(64, 10);
			df.display(gw, DisplayOp().delay(5000));
		}
		else if (FLAGS_x17) {			
			auto imba = df.imbatch(FLAGS_celeba128, { 50, 3, 128, 128 });
			auto gb = df.gaussian_blur(imba, 63, 10);
			df.display(gb, DisplayOp().delay(5000));
		}
		else if (FLAGS_x18) { 			
			auto input = df.imbatch(FLAGS_celeba128, { 40, 3, 128, 128 });
			auto patches = df.patch_sampling(input, 64, 32);
			df.display(patches, DisplayOp().delay(5000));
		}
		else if (FLAGS_x19) {
			// rotation and flip
			auto image = df.imread(FLAGS_image1, ImreadOp().color());
			auto theta = df.variable(df.constant({ 1, 1, 2, 3 }, { -1, 0, 0, 0, -1, 0}));
			auto sp = df.spatial_transformer(image, theta, 1, 256, 256, "");
			df.display(sp, DisplayOp().delay(5000));
		}
		else if (FLAGS_x20) {
			// rotation and flip
			auto image = df.imbatch(FLAGS_celeba128, { 1, 3, 128, 128 });
			auto theta = df.variable(df.constant({ 2, 1, 2, 3 }, 
			{ -1, 0, 0, 0,-1, 0,
			   1, 0, 0, 0,-1, 0 }));
			auto sp = df.spatial_transformer(image, theta, 2, 128, 128, "");
			df.display(sp, DisplayOp().delay(5000));
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

	auto execution_context = std::make_shared<ExecutionContext>();
	execution_context->debug_level = FLAGS_debug;

	session->initialize(execution_context);		
	auto psnr_node = session->get_node<Psnr>("psnr", "", false);
	for (int iter = 1; iter <= FLAGS_iter; ++iter) {		
		std::cout << "Iteration " << iter << " -->" << std::endl;
		auto epoch_start = std::chrono::high_resolution_clock::now();
		session->forward("");
		if (psnr_node) {
			std::cout << "PSNR: " << psnr_node->psnr() << std::endl;
		}
		session->backward("");
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


