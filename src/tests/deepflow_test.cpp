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
		df.variable(df.fill({ 3,3,3,3 }, value), "", VariableOp("var"));
		auto session = df.session();
		session->initialize();
		auto var = session->get_node("var");
		session->forward({ var });		
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
		df.variable(df.index_fill({ 3,3,3,3 }, value), "", VariableOp("var"));
		auto session = df.session();
		session->initialize();
		auto var = session->get_node("var");
		session->forward({var});
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
		df.variable(df.random_uniform({ 3,3,3,3 }, min_value, max_value), "", VariableOp("var"));
		auto session = df.session();
		session->initialize();
		auto var = session->get_node("var");
		session->forward({ var });
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
		df.variable(df.step({ 3,3,3,3 }, min_value, max_value), "", VariableOp("var"));
		auto session = df.session();
		session->initialize();
		auto var = session->get_node("var");
		session->forward({ var });
		double min, max, avg, std;
		var->output(0)->value()->statistics(&avg, &std, &min, &max);
		EXPECT_GE(min, min_value);
		EXPECT_LE(max, max_value);
		EXPECT_NE(min, max);
	}
}

TEST(abs, forward) {
	DeepFlow df;
	auto node = df.variable(df.step({ 3,3,3,3 }, -100, 100), "", VariableOp("var"));
	df.abs(node, AbsOp("abs"));
	auto session = df.session();
	session->initialize();
	auto abs = session->get_node("abs");
	session->forward({ abs });
	double min, max, avg, std;
	abs->output(0)->value()->statistics(&avg, &std, &min, &max);
	EXPECT_GE(min, 0);
	EXPECT_LE(max, 100);
	EXPECT_NE(min, max);
}

TEST(bias_add, forward) {
	DeepFlow df;
	auto vec = df.place_holder({ 2, 3, 2, 2 }, PlaceholderOp("v"));
	auto bias = df.place_holder({ 1, 3, 1, 1 }, PlaceholderOp("b"));
	df.bias_add(vec, bias, BiasAddOp("bias_add"));
	auto session = df.session();
	session->initialize();
	session->get_node("v")->write_values({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->get_node("b")->write_values({ 1, 2, 3 });
	auto bias_add_node = session->get_node("bias_add");
	session->forward({ bias_add_node });
	EXPECT_EQ(
		bias_add_node->output(0)->value()->verify({ 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 14, 15, 16, 17, 19, 20, 21, 22, 24, 25, 26, 27 }),
		true);
}

TEST(bias_add, backward) {
	DeepFlow df;
	auto vec = df.place_holder({ 2, 3, 2, 2 },  PlaceholderOp("v"));
	auto bias = df.place_holder({ 1, 3, 1, 1 }, PlaceholderOp("b"));
	df.bias_add(vec, bias, BiasAddOp("bias_add"));
	auto session = df.session();
	session->initialize();	
	session->get_node("v")->write_values({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->get_node("b")->write_values({ 1, 2, 3 });
	auto bias_add_node = session->get_node("bias_add");
	session->forward({ bias_add_node });
	bias_add_node->write_diffs({
		24, 23, 22, 21,
		20, 19, 18, 17,
		16, 15, 14, 13,
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1 });
	session->backward({ bias_add_node });
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
	auto input = df.place_holder({ 2, 1, 3, 3 }, PlaceholderOp("input"));
	auto f = df.place_holder({ 1, 1, 2, 2 }, PlaceholderOp("f"));
	df.conv2d(input, f, ConvolutionOp("conv").pad(0));
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
	auto conv = session->get_node("conv");
	session->forward({ conv });
	EXPECT_EQ(
		conv->output(0)->value()->verify({ 
		12, 16,
		24, 28,
		
		48, 52,
		60, 64 }),
		true);
}

TEST(add, forward) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, PlaceholderOp("a"));
	auto b = df.place_holder({ 2, 3, 2, 2 }, PlaceholderOp("b"));
	df.add(a, b, AddOp("add"));
	auto session = df.session();
	session->initialize();
	session->get_node("a")->write_values({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->get_node("b")->write_values({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	auto add_node = session->get_node("add");
	session->forward({add_node});
	EXPECT_EQ(add_node->output(0)->value()->verify({ 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48 }), true);
}

TEST(sub, forward) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, PlaceholderOp("a"));
	auto b = df.place_holder({ 2, 3, 2, 2 }, PlaceholderOp("b"));
	df.subtract(a, b, SubtractOp("sub"));
	auto session = df.session();
	session->initialize();
	session->get_node("a")->write_values({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25 });
	session->get_node("b")->write_values({ 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	auto sub_node = session->get_node("sub");
	session->forward({sub_node});
	EXPECT_EQ(sub_node->output(0)->value()->verify({ -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }), true);
}

TEST(add, backward) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, PlaceholderOp("a"));
	auto b = df.place_holder({ 2, 3, 2, 2 }, PlaceholderOp("b"));
	df.add(a, b, AddOp("add"));
	auto session = df.session();
	session->initialize();
	session->get_node("add")->write_diffs({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->backward({ session->get_node("add") });
	EXPECT_EQ(session->get_node("a")->output(0)->diff()->verify({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 }), true);
	EXPECT_EQ(session->get_node("b")->output(0)->diff()->verify({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 }), true);
}

TEST(sub, backward) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, PlaceholderOp("a"));
	auto b = df.place_holder({ 2, 3, 2, 2 }, PlaceholderOp("b"));
	df.subtract(a, b, SubtractOp("sub"));
	auto session = df.session();
	session->initialize();
	session->get_node("sub")->write_diffs({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->backward({ session->get_node("sub") });
	EXPECT_EQ(session->get_node("a")->output(0)->diff()->verify({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 }), true);
	EXPECT_EQ(session->get_node("b")->output(0)->diff()->verify({ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24 }), true);
}

TEST(batch_stddev, forward) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, PlaceholderOp("a"));
	df.batch_stddev(a, BatchStddevOp("stddev"));
	auto session = df.session();
	session->initialize();
	session->get_node("a")->write_values({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25 });
	session->forward({session->get_node("stddev")});
	auto stddev = session->get_node("stddev");
	LOG(INFO) << stddev->output(0)->value()->toString();
}

TEST(leaky_relu, forward_zero) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, PlaceholderOp("a"));
	df.leaky_relu(a, LeakyReluOp("relu").negative_slope(0));
	auto session = df.session();
	session->initialize();
	session->get_node("a")->write_values({ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->forward({session->get_node("relu")});
	EXPECT_EQ(session->get_node("relu")->output(0)->value()->verify({ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 }), true);	
}

TEST(leaky_relu, forward_1) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, PlaceholderOp("a"));
	df.leaky_relu(a, LeakyReluOp("relu").negative_slope(1.0f));
	auto session = df.session();
	session->initialize();
	session->get_node("a")->write_values({ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->forward({ session->get_node("relu") });
	EXPECT_EQ(session->get_node("relu")->output(0)->value()->verify({ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 }), true);
}

TEST(leaky_relu, forward_0_1) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, PlaceholderOp("a"));
	df.leaky_relu(a, LeakyReluOp("relu").negative_slope(0.1f));
	auto session = df.session();
	session->initialize();
	session->get_node("a")->write_values({ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->forward({ session->get_node("relu") });
	EXPECT_EQ(session->get_node("relu")->output(0)->value()->verify({ -0.1f, -0.2f, -0.3f, -0.4f, -0.5f, -0.6f, -0.7f, -0.8f, -0.9f, -1.0f, -1.1f, -1.2f, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 }), true);
}

TEST(leaky_relu, backward_zero) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, PlaceholderOp("a"));
	df.leaky_relu(a, LeakyReluOp("relu").negative_slope(0));
	auto session = df.session();
	session->initialize();
	session->get_node("a")->write_values({ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->forward({ session->get_node("relu") });
	session->get_node("relu")->write_diffs({ -2, -4, -6, -8, -10, -12, -14, -16, -18, -20, -22, -24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48 });
	session->backward({ session->get_node("relu") });
	EXPECT_EQ(session->get_node("a")->output(0)->diff()->verify({ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48 }), true);
}

TEST(leaky_relu, backward_one) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, PlaceholderOp("a"));
	df.leaky_relu(a, LeakyReluOp("relu").negative_slope(1.0f));
	auto session = df.session();
	session->initialize();
	session->get_node("a")->write_values({ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->forward({ session->get_node("relu") });
	session->get_node("relu")->write_diffs({ -2, -4, -6, -8, -10, -12, -14, -16, -18, -20, -22, -24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48 });
	session->backward({ session->get_node("relu") });
	EXPECT_EQ(session->get_node("a")->output(0)->diff()->verify({ -2, -4, -6, -8, -10, -12, -14, -16, -18, -20, -22, -24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48 }), true);
	EXPECT_EQ(session->get_node("a")->output(0)->value()->verify({ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 }), true);
}

TEST(leaky_relu, backward_0_1) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, PlaceholderOp("a"));
	df.leaky_relu(a, LeakyReluOp("relu").negative_slope(0.1f));
	auto session = df.session();
	session->initialize();
	session->get_node("a")->write_values({ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
	session->forward({ session->get_node("relu") });
	session->get_node("relu")->write_diffs({ -2, -4, -6, -0.000008f, -10, -12, -14, -16000, -18, -20, -22, -24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48 });
	session->backward({ session->get_node("relu") });
	EXPECT_EQ(session->get_node("a")->output(0)->diff()->verify({ -0.2f, -0.4f, -0.6f, -0.0000008f, -1.0f, -1.2f, -1.4f, -1600.0f, -1.8f, -2.0f, -2.2f, -2.4f, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48 }), true);
	EXPECT_EQ(session->get_node("a")->output(0)->value()->verify({ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 }), true);
}

TEST(gradient_fill, fill_test) {
	DeepFlow df;
	df.variable(df.gradient({ 3, 2 , 8 , 8 }), "", VariableOp("var"));
	auto session = df.session();
	session->initialize();
	auto var = session->get_node("var");
	session->forward({ var });
	double min, max, avg, std;
	var->output(0)->value()->statistics(&avg, &std, &min, &max);	
	EXPECT_LE(min, -1);
	EXPECT_GE(max, 1);	
}

TEST(concate, forward1) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, PlaceholderOp("a"));
	auto b = df.place_holder({ 2, 3, 2, 2 }, PlaceholderOp("b"));
	df.concate(a, b, ConcateOp("concate"));
	auto session = df.session();
	session->initialize();
	session->get_node("a")->write_values({  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24 });
	session->get_node("b")->write_values({ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16 });
	auto concate = session->get_node("concate");
	session->forward({ concate });	
	EXPECT_EQ(concate->output(0)->value()->verify({ 1,2,3,4,5,6,7,8,9,10,11,12,-1,-2,-3,-4,-5,-6,-7,-8,13,14,15,16,17,18,19,20,21,22,23,24,-9,-10,-11,-12,-13,-14,-15,-16 }), true);
}

TEST(softmax, forward) {
	DeepFlow df;
	auto a = df.place_holder({ 2, 3, 2, 2 }, PlaceholderOp("a"));
	df.softmax(a, SoftmaxOp("softmax1").by_instance());
	df.softmax(a, SoftmaxOp("softmax2").by_channel());
	auto session = df.session();
	session->initialize();
	session->get_node("a")->write_values({ 0.1f,  0.2f,  0.3f,  0.4f,  0.5f,  0.6f,  0.7f,  0.8f,  0.9f,  0.10f,  0.11f,  0.12f,  0.13f,  0.14f,  0.15f,  0.16f,  0.17f,  0.18f,  0.19f,  0.20f,  0.21f,  0.22f,  0.23f,  0.24f });
	auto softmax1 = session->get_node("softmax1");
	session->forward({ softmax1 });	
	// 0.059141 + 0.065361 + 0.072235 + 0.079832 + 0.088229 + 0.097508 + 0.107763 + 0.119096 + 0.131622 + 0.059141 + 0.059736 + 0.060336 = 1
	EXPECT_EQ(softmax1->output(0)->value()->verify(
	{ 0.059141f, 0.065361f, 0.072235f, 0.079832f, 0.088229f, 0.097508f, 0.107763f, 0.119096f, 0.131622f, 0.059141f, 0.059736f, 0.060336f, 0.078827f, 0.079619f, 0.080419f, 0.081227f, 0.082044f, 0.082868f, 0.083701f, 0.084542f, 0.085392f, 0.086250f, 0.087117f, 0.087993f }), true);

	auto softmax2 = session->get_node("softmax2");
	session->forward({ softmax2 });	
	// 0.211983 + 0.316241 + 0.471776 = 1
	EXPECT_EQ(softmax2->output(0)->value()->verify(
	{ 0.211983f, 0.294407f, 0.301315f, 0.307919f, 0.316241f, 0.439203f, 0.449509f, 0.459361f, 0.471776f, 0.266390f, 0.249175f, 0.232720f, 0.320092f, 0.320092f, 0.320092f, 0.320092f, 0.333156f, 0.333156f, 0.333156f, 0.333156f, 0.346752f, 0.346752f, 0.346752f, 0.346752f }), true);

}

int main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);	
	CudaHelper::setOptimalThreadsPerBlock();
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}