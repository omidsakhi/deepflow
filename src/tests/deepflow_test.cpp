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
}

TEST(conv2d, forward) {
	DeepFlow df;
	auto input = df.place_holder({ 2, 1, 3, 3 }, Tensor::Float, "input");
	auto f = df.place_holder({ 1, 1, 2, 2 }, Tensor::Float, "f");
	auto conv = df.conv2d(input, f, 0, 0, 1, 1, 1, 1, "conv");
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

int main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);	
	CudaHelper::setOptimalThreadsPerBlock();
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}