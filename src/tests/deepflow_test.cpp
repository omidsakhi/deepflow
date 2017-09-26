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

int main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}