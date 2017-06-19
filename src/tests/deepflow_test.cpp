#include "core/deep_flow.h"
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "core/session.h"

TEST(InitializerTest, FillTest) {
	DeepFlow df;
	df.variable(df.fill({ 1,1,1,1 }, 2), "", "var1");
	df.variable(df.fill({ 1,1,1,1 }, -1), "", "var2");
	auto session = df.session();
	session->initialize();
	session->forward();
	auto var1 = session->get_node("var1");
	auto var2 = session->get_node("var2");
	EXPECT_EQ(var1->output(0)->value()->toFloat(), 2);	
	EXPECT_EQ(var2->output(0)->value()->toFloat(), -1);
}

TEST(BatchNormalization, LoadAndSave) {
	DeepFlow df;
	
	auto var = df.variable(df.random_normal({ 32, 32, 28, 28 }, 0, 0.2), "", "var");
	auto bn = df.batch_normalization(var, DeepFlow::PER_ACTIVATION, 0, 1, 0, 1, 1);

	auto session = df.session();
	session->initialize();

	for (int i = 1; i < 3; i++) 
	{		
		session->forward();
		session->save("temp.bin");
	}
}

int main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}