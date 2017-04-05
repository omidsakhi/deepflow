
#include "core/deep_flow.h"

void main() {
	CudaHelper::setOptimalThreadsPerBlock();

	int batch_size = 100;
	DeepFlow df;
	auto mnist_trainset = df.mnist_reader("./data/mnist", batch_size, MNISTReaderType::Train);
	auto mnist_testset = df.mnist_reader("./data/mnist", batch_size, MNISTReaderType::Test);
	
	auto x = df.place_holder({ 100, 1, 28, 28 }, Tensor::TensorType::Float, "x");
	auto y = df.place_holder({ 100, 10, 1, 1 }, Tensor::TensorType::Float, "y");
	auto f1 = df.variable(df.random_uniform({ 20, 1 , 5, 5 }, -0.1f, 0.1f), "f1");
	auto conv1 = df.conv2d(x, f1, "conv1");
	auto pool1 = df.pooling(conv1, 2, 2, 0, 0, 2, 2, "pool1");
	auto f2 = df.variable(df.random_uniform({ 50, 20 , 5, 5 }, -0.1f, 0.1f), "f2");
	auto conv2 = df.conv2d(pool1, f2, "conv2");
	auto pool2 = df.pooling(conv2, 2, 2, 0, 0, 2, 2, "pool2");
	auto w1 = df.variable(df.random_uniform({ 800, 500, 1 , 1 }, -0.1f, 0.1f), "w1");
	auto b1 = df.variable(df.step({ 1, 500, 1 , 1 }, -1.0f, 1.0f), "b1");
	auto a1 = df.relu(df.bias_add(df.matmul(pool2, w1), b1));
	auto d1 = df.dropout(a1);
	auto w2 = df.variable(df.random_uniform({ 500, 10, 1 , 1 }, -0.1f, 0.1f), "w2");
	auto b2 = df.variable(df.step({ 1, 10, 1, 1 }, -1.0f, 1.0f), "b2");
	auto a2 = df.relu(df.bias_add(df.matmul(d1, w2), b2));
	auto loss = df.softmax_loss(a2, y);
	auto mse = df.reduce_mean(df.reduce_mean(df.square(loss),1),0);
	auto accuracy = df.reduce_mean(
		df.equal(
			df.argmax(loss->node()->output(1), 1),
			df.argmax(y, 1)),0);

	df.global_node_initializer();

	auto trainer = df.gain_solver(loss, 2000, 0.9999f, 0.0001f, 100, 0.1, 0.05, 0.95);
	int epoch = 1;
	for (int iteration = 0; iteration < 10000; ++iteration) {
		x->feed(mnist_trainset->output(0));
		y->feed(mnist_trainset->output(1));
		trainer->train_step();
		//df.eval(mse);		
		//std::cout << "Iteraration " << iteration << " - MSE: " << mse->value()->toFloat() << std::endl;
		if (mnist_trainset->isLastBatch()) {
			double sum = 0;
			size_t count = 0;
			do {
				x->feed(mnist_testset->output(0));
				y->feed(mnist_testset->output(1));
				df.eval(accuracy);
				sum += accuracy->value()->toFloat();
				count++;
				mnist_testset->nextBatch();
			} while (!mnist_testset->isLastBatch());
			std::cout << "Epoch " << epoch << " Accuracy: " << sum / count << std::endl;
			epoch++;
		}
		mnist_trainset->nextBatch();
	}	
}

