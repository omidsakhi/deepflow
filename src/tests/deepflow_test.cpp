#include "core/deep_flow.h"
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <random>
#include "core/session.h"

TEST(fill, initialization) {
	std::random_device r;
	std::default_random_engine e1(r());
	std::uniform_int_distribution<int> uniform_dist(0, 10000);

	for (int i = 0; i < 10; ++i) {
		int value = uniform_dist(e1);
		DeepFlow df;
		df.variable(df.fill({ 3,3,3,3 }, value), "", "var");
		auto session = df.session();
		session->initialize();
		session->forward();
		auto var = session->get_node("var");
		double min, max, avg, std;
		var->output(0)->value()->statistics(&avg, &std, &min, &max);
		EXPECT_EQ(min, value);
		EXPECT_EQ(max, value);
		EXPECT_EQ(avg, value);
	}
}

TEST(index_fill, initialization) {
	std::random_device r;
	std::default_random_engine e1(r());
	std::uniform_int_distribution<int> uniform_dist(0, 10000);

	for (int i = 0; i < 10; ++i) {
		int value = uniform_dist(e1);
		DeepFlow df;
		df.variable(df.index_fill({ 3,3,3,3 }, value), "", "var");
		auto session = df.session();
		session->initialize();
		session->forward();
		auto var = session->get_node("var");
		double min, max, avg, std;
		var->output(0)->value()->statistics(&avg, &std, &min, &max);
		EXPECT_EQ(min, value);
		EXPECT_EQ(max, value + 80);
		double sum = 0;
		for (int k = value; k < value + 81; ++k)
			sum += k;
		EXPECT_EQ(avg, sum / 81);
	}
}

TEST(random_uniform, initialization) {
	std::random_device r;
	std::default_random_engine e1(r());
	std::uniform_int_distribution<int> uniform_dist(0, 10000);

	for (int i = 0; i < 10; ++i) {
		float min_value = uniform_dist(e1);
		float max_value = min_value + uniform_dist(e1);		
		DeepFlow df;
		df.variable(df.random_uniform({ 3,3,3,3 }, min_value, max_value), "", "var");
		auto session = df.session();
		session->initialize();
		session->forward();
		auto var = session->get_node("var");
		double min, max, avg, std;
		var->output(0)->value()->statistics(&avg, &std, &min, &max);
		EXPECT_GE(min, min_value);
		EXPECT_LE(max, max_value);
		EXPECT_NE(min, max);
	}
}

TEST(step, initialization) {
	std::random_device r;
	std::default_random_engine e1(r());
	std::uniform_int_distribution<int> uniform_dist(0, 10000);

	for (int i = 0; i < 10; ++i) {
		float min_value = uniform_dist(e1);
		float max_value = min_value + uniform_dist(e1);		
		DeepFlow df;
		df.variable(df.step({ 3,3,3,3 }, min_value, max_value), "", "var");
		auto session = df.session();
		session->initialize();
		session->forward();
		auto var = session->get_node("var");
		double min, max, avg, std;
		var->output(0)->value()->statistics(&avg, &std, &min, &max);
		EXPECT_GE(min, min_value);
		EXPECT_LE(max, max_value);
		EXPECT_NE(min, max);
	}
}

TEST(abs, forward) {
	DeepFlow df;
	auto node = df.variable(df.step({ 3,3,3,3 }, -100, 100), "", "var");
	df.abs(node, "abs");
	auto session = df.session();
	session->initialize();
	session->forward();
	auto abs = session->get_node("abs");
	double min, max, avg, std;
	abs->output(0)->value()->statistics(&avg, &std, &min, &max);
	EXPECT_GE(min, 0);
	EXPECT_LE(max, 100);
	EXPECT_NE(min, max);
}

TEST(bias_add, forward) {
	DeepFlow df;
	auto vec = df.place_holder({ 2, 3, 2, 2 }, Tensor::Float, "v");
	auto bias = df.place_holder({ 1, 3, 1, 1 }, Tensor::Float, "b");
	auto bias_add_op = df.bias_add(vec, bias, "bias_add");
	auto session = df.session();
	session->initialize();
	session->get_node("v")->write_values({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->get_node("b")->write_values({ 1, 2, 3 });
	session->forward();	
	EXPECT_EQ(
		session->get_node("bias_add")->output(0)->value()->verify({ 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 14, 15, 16, 17, 19, 20, 21, 22, 24, 25, 26, 27 }),
		true);
}

TEST(bias_add, backward) {
	DeepFlow df;
	auto vec = df.place_holder({ 2, 3, 2, 2 }, Tensor::Float, "v");
	auto bias = df.place_holder({ 1, 3, 1, 1 }, Tensor::Float, "b");
	auto bias_add_op = df.bias_add(vec, bias, "bias_add");
	auto session = df.session();
	session->initialize();	
	session->get_node("v")->write_values({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->get_node("b")->write_values({ 1, 2, 3 });
	session->forward();	
	session->get_node("bias_add")->write_diffs({ 
		24, 23, 22, 21,
		20, 19, 18, 17,
		16, 15, 14, 13,
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1 });
	session->backward();
	EXPECT_EQ(session->get_node("bias_add")->input(0)->diff()->verify({ 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }), true);	
	EXPECT_EQ(session->get_node("bias_add")->input(1)->diff()->verify({ 11.75, 9.75, 7.75 }), true);	
	EXPECT_EQ(
		session->get_node("bias_add")->output(0)->value()->verify({ 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 14, 15, 16, 17, 19, 20, 21, 22, 24, 25, 26, 27 }),
		true);
	EXPECT_EQ(session->get_node("v")->output(0)->value()->verify({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 }), true);
	EXPECT_EQ(session->get_node("b")->output(0)->value()->verify({ 1, 2, 3 }), true);
}

TEST(conv2d, forward) {
	DeepFlow df;
	auto input = df.place_holder({ 2, 1, 3, 3 }, Tensor::Float, "input");
	auto f = df.place_holder({ 1, 1, 2, 2 }, Tensor::Float, "f");
	auto conv = df.conv2d(input, f, "", 0, 0, 0, 1, 1, 1, 1, "conv");
	auto session = df.session();
	session->initialize();
	session->get_node("input")->write_values(
	{ 1, 2, 3, 
	  4, 5, 6,
	  7, 8, 9,
	  
	  10, 11, 12,
	  13, 14, 15,
	  16, 17, 18 });
	session->get_node("f")->write_values({ 1, 1, 1, 1 });
	session->forward();	
	EXPECT_EQ(
		session->get_node("conv")->output(0)->value()->verify({ 
		12, 16,
		24, 28,
		
		48, 52,
		60, 64 }),
		true);
}

TEST(add, forward) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, Tensor::Float, "a");
	auto b = df.place_holder({ 2, 3, 2, 2 }, Tensor::Float, "b");
	auto add_op = df.add(a, b, "add");
	auto session = df.session();
	session->initialize();
	session->get_node("a")->write_values({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->get_node("b")->write_values({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->forward();	
	EXPECT_EQ(session->get_node("add")->output(0)->value()->verify({ 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48 }), true);
}

TEST(sub, forward) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, Tensor::Float, "a");
	auto b = df.place_holder({ 2, 3, 2, 2 }, Tensor::Float, "b");
	auto add_op = df.add(a, b, 1.0, -1.0, "sub");
	auto session = df.session();
	session->initialize();
	session->get_node("a")->write_values({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25 });
	session->get_node("b")->write_values({ 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->forward();
	EXPECT_EQ(session->get_node("sub")->output(0)->value()->verify({ -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }), true);
}

TEST(add, backward) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, Tensor::Float, "a");
	auto b = df.place_holder({ 2, 3, 2, 2 }, Tensor::Float, "b");
	auto add_op = df.add(a, b, "add");
	auto session = df.session();
	session->initialize();
	session->get_node("add")->write_diffs({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->backward();
	EXPECT_EQ(session->get_node("a")->output(0)->diff()->verify({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 }), true);
	EXPECT_EQ(session->get_node("b")->output(0)->diff()->verify({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 }), true);
}

TEST(sub, backward) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, Tensor::Float, "a");
	auto b = df.place_holder({ 2, 3, 2, 2 }, Tensor::Float, "b");
	auto add_op = df.add(a, b, 1.0, -1.0, "sub");
	auto session = df.session();
	session->initialize();
	session->get_node("sub")->write_diffs({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->backward();
	EXPECT_EQ(session->get_node("a")->output(0)->diff()->verify({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 }), true);
	EXPECT_EQ(session->get_node("b")->output(0)->diff()->verify({ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24 }), true);
}

TEST(batch_stddev, forward) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, Tensor::Float, "a");	
	df.batch_stddev(a, "stddev");
	auto session = df.session();
	session->initialize();
	session->get_node("a")->write_values({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25 });
	session->forward();
	auto stddev = session->get_node("stddev");
	LOG(INFO) << stddev->output(0)->value()->toString();
}

TEST(leaky_relu, forward_zero) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, Tensor::Float, "a");	
	auto relu_op = df.leaky_relu(a, 0, "relu");
	auto session = df.session();
	session->initialize();
	session->get_node("a")->write_values({ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->forward();
	EXPECT_EQ(session->get_node("relu")->output(0)->value()->verify({ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 }), true);	
}

TEST(leaky_relu, forward_1) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, Tensor::Float, "a");
	auto relu_op = df.leaky_relu(a, 1.0, "relu");
	auto session = df.session();
	session->initialize();
	session->get_node("a")->write_values({ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->forward();
	EXPECT_EQ(session->get_node("relu")->output(0)->value()->verify({ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 }), true);
}

TEST(leaky_relu, forward_0_1) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, Tensor::Float, "a");
	auto relu_op = df.leaky_relu(a, 0.1, "relu");
	auto session = df.session();
	session->initialize();
	session->get_node("a")->write_values({ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->forward();	
	EXPECT_EQ(session->get_node("relu")->output(0)->value()->verify({ -0.1f, -0.2f, -0.3f, -0.4f, -0.5f, -0.6f, -0.7f, -0.8f, -0.9f, -1.0f, -1.1f, -1.2f, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 }), true);
}

TEST(leaky_relu, backward_zero) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, Tensor::Float, "a");
	auto relu_op = df.leaky_relu(a, 0, "relu");
	auto session = df.session();
	session->initialize();
	session->get_node("a")->write_values({ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->forward();
	session->get_node("relu")->write_diffs({ -2, -4, -6, -8, -10, -12, -14, -16, -18, -20, -22, -24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48 });
	session->backward();
	EXPECT_EQ(session->get_node("a")->output(0)->diff()->verify({ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48 }), true);
}

TEST(leaky_relu, backward_one) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, Tensor::Float, "a");
	auto relu_op = df.leaky_relu(a, 1.0, "relu");
	auto session = df.session();
	session->initialize();
	session->get_node("a")->write_values({ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->forward();
	session->get_node("relu")->write_diffs({ -2, -4, -6, -8, -10, -12, -14, -16, -18, -20, -22, -24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48 });
	session->backward();
	EXPECT_EQ(session->get_node("a")->output(0)->diff()->verify({ -2, -4, -6, -8, -10, -12, -14, -16, -18, -20, -22, -24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48 }), true);
	EXPECT_EQ(session->get_node("a")->output(0)->value()->verify({ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 }), true);
}

TEST(leaky_relu, backward_0_1) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, Tensor::Float, "a");
	auto relu_op = df.leaky_relu(a, 0.1f, "relu");
	auto session = df.session();
	session->initialize();
	session->get_node("a")->write_values({ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->forward();
	session->get_node("relu")->write_diffs({ -2, -4, -6, -0.000008f, -10, -12, -14, -16000, -18, -20, -22, -24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48 });
	session->backward();
	EXPECT_EQ(session->get_node("a")->output(0)->diff()->verify({ -0.2f, -0.4f, -0.6f, -0.0000008f, -1.0f, -1.2f, -1.4f, -1600.0f, -1.8f, -2.0f, -2.2f, -2.4f, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48 }), true);
	EXPECT_EQ(session->get_node("a")->output(0)->value()->verify({ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 }), true);
}

TEST(gradient_fill, fill_test) {
	DeepFlow df;
	df.variable(df.gradient({ 3, 2 , 8 , 8 }), "", "var");
	auto session = df.session();
	session->initialize();
	session->forward();
	auto var = session->get_node("var");
	double min, max, avg, std;
	var->output(0)->value()->statistics(&avg, &std, &min, &max);	
	EXPECT_LE(min, -1);
	EXPECT_GE(max, 1);	
}

int main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);	
	CudaHelper::setOptimalThreadsPerBlock();
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}