#include "generators/mnist_reader.h"

#include <glog/logging.h>

MNISTReader::MNISTReader(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_mnist_param() == false) << "param.has_mnist_param() == false";
	auto mnist_param = param->mnist_param();	
	_folder_path = mnist_param.folder_path();
	_batch_size = mnist_param.batch_size();
	_reader_type = (MNISTReaderType)mnist_param.reader_type();	
	_output_type = (MNISTOutputType)mnist_param.output_type();	
	if (_reader_type == MNISTReaderType::Train)
		_num_total_samples = 60000;
	else
		_num_total_samples = 10000;
	_num_batches = _num_total_samples / _batch_size;
	_last_batch = (_current_batch == (_num_batches - 1));
}

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void MNISTReader::initForward() {
	_file_path = _folder_path;
	if (_output_type == MNISTOutputType::Data && _reader_type == MNISTReaderType::Train) {
		_file_path += "/train-images.idx3-ubyte";
	}
	else if (_output_type == MNISTOutputType::Data && _reader_type == MNISTReaderType::Test) {
		_file_path += "/t10k-images.idx3-ubyte";
	} 
	else if (_output_type == MNISTOutputType::Labels && _reader_type == MNISTReaderType::Train) {
		_file_path += "/train-labels.idx1-ubyte";
	} 
	else if (_output_type == MNISTOutputType::Labels && _reader_type == MNISTReaderType::Test) {			
		_file_path += "/t10k-labels.idx1-ubyte";
	}
	else {
		LOG(FATAL);
	}

	LOG(INFO) << "MNIST " << _name << " reading from " << _file_path;

	if (_output_type == MNISTOutputType::Data) {
		_tx.open(_file_path, std::ios::binary);
		LOG_IF(FATAL, !_tx.is_open()) << "TX not open.";
		int msb = 0;
		_tx.read((char*)&msb, sizeof(__int32));
		int n = 0;
		_tx.read((char*)&n, sizeof(__int32));
		n = ReverseInt(n);
		LOG_IF(FATAL, n != _num_total_samples) << "TX error.";
		int n_rows, n_cols;
		_tx.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		LOG_IF(FATAL, n_rows != 28) << "Rows not equal to 28.";
		_tx.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		LOG_IF(FATAL, n_cols != 28) << "Columns not equal to 28.";
		_tx_start_pos = _tx.tellg();
		_buf = new float[_batch_size * 28 * 28];
		_temp = new unsigned char[28 * 28];
		_outputs[0]->initValue({ _batch_size, 1, 28, 28 });
	}
	else if (_output_type == MNISTOutputType::Labels) {
		_tx.open(_file_path, std::ios::binary);
		LOG_IF(FATAL, !_tx.is_open()) << "Failed to open " << _file_path;
		int msb = 0;
		_tx.read((char*)&msb, sizeof(__int32));
		int n = 0;
		_tx.read((char*)&n, sizeof(__int32));
		n = ReverseInt(n);
		LOG_IF(FATAL, n != _num_total_samples) << _file_path << " error. " << n;
		_tx_start_pos = _tx.tellg();
		_buf = new float[_batch_size * 10];		
		_temp = new unsigned char[1];
		_outputs[0]->initValue({ _batch_size, 10, 1, 1 });
	}
	else {
		LOG(FATAL);
	}	
}

void MNISTReader::initBackward() {
	
}

void MNISTReader::forward()
{
	_last_batch = (_current_batch >= (_num_batches - 1));
	if (_last_batch == true)
	{
		_tx.clear(); // Reached EOF. Must reset the flag.
		_tx.seekg(_tx_start_pos);
		_current_batch = 0;
	}
	else
	{
		_current_batch++;
	}

	if (_output_type == MNISTOutputType::Data) {
		for (int i = 0; i < _batch_size; ++i)
		{
			_tx.read((char*)_temp, 28 * 28);
			for (int j = 0; j < 28 * 28; ++j)
				_buf[i * 28 * 28 + j] = ((float)_temp[j] / 255.0f - 0.5f) * 2;
		}
	}
	else if (_output_type == MNISTOutputType::Labels) {
		for (int i = 0; i < _batch_size * 10; i++)
			_buf[i] = 0.0f;
		for (int i = 0; i < _batch_size; ++i)
		{
			_tx.read((char*)_temp, 1);
			_buf[i * 10 + _temp[0]] = 1.0f;
		}
		/*
		for (int i = 0; i < _num_batch_samples; ++i) {
		for (int j = 0; j < 10; ++j)
		{
		std::cout << _buf[i * 10 + j] << " ";
		}
		std::cout << ", ";
		}
		std::cout << std::endl;
		*/
	}
	else {
		LOG(FATAL);
	}

	DF_NODE_CUDA_CHECK(cudaMemcpy(_outputs[0]->value()->mutableData(), _buf, _outputs[0]->value()->sizeInBytes(), cudaMemcpyHostToDevice));
	LOG_IF(INFO, _verbose > 2) << "MNIST " << _name << " - BATCH @ " << _current_batch;
}

void MNISTReader::deinit() {
	if (_temp) delete[] _temp;
	if (_buf) delete[] _buf;	
	_tx.close();	
}

bool MNISTReader::isLastBatch() {
	return _last_batch;
}

std::string MNISTReader::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.mnist_reader(\"" + _folder_path + "\", ";
	cpp += std::to_string(_batch_size) + ", ";
	if (_reader_type == Train)
		cpp += "MNISTReader::Train, ";
	else 
		cpp += "MNISTReader::Test, ";
	if (_output_type == Data)
		cpp += "MNISTReader::Data, ";
	else 
		cpp += "MNISTReader::Labels, ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
