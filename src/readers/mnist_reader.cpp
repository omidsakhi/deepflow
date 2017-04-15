#include "readers/mnist_reader.h"

#include <glog/logging.h>

MNISTReader::MNISTReader(const NodeParam &param) : Reader(param) {
	LOG_IF(FATAL, param.reader_param().has_mnist_param() == false) << "param.has_mnist_reader_param() == false";
	const MnistReaderParam &mnist_param = param.reader_param().mnist_param();
	_folder_path = mnist_param.folder_path();
	_batch_size = mnist_param.batch_size();
	_type = (MNISTReaderType)mnist_param.type();	
	_temp_d = 0;	
	_init_backward = _init_backward = false;
	if (_type == MNISTReaderType::Train)
		_num_total_samples = 60000;
	else
		_num_total_samples = 10000;
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
	std::string data_file = _folder_path;
	if (_type == MNISTReaderType::Train)
		data_file += "/train-images.idx3-ubyte";
	else
		data_file += "/t10k-images.idx3-ubyte";
	LOG(INFO) << "Initializing data from " << data_file;
	_tx.open(data_file, std::ios::binary);
	LOG_IF(FATAL, !_tx.is_open()) << "TX not open.";
	int msb = 0;
	_tx.read((char*)&msb, sizeof(__int32));
	int n = 0;
	_tx.read((char*)&n, sizeof(__int32));
	n = ReverseInt(n);
	LOG_IF(FATAL, n != 60000 && n != 10000) << "TX error.";
	int n_rows, n_cols;
	_tx.read((char*)&n_rows, sizeof(n_rows));
	n_rows = ReverseInt(n_rows);
	LOG_IF(FATAL, n_rows != 28) << "Rows not equal to 28.";
	_tx.read((char*)&n_cols, sizeof(n_cols));
	n_cols = ReverseInt(n_cols);
	LOG_IF(FATAL, n_cols != 28) << "Columns not equal to 28.";
	_tx_start_pos = _tx.tellg();
	_data_buf = new float[_batch_size * 28 * 28];
	_temp_d = new unsigned char[28 * 28];	
	_outputs[0]->initValue({ _batch_size, 1, 28, 28 });	
	_init_forward = true;

	std::string label_file = _folder_path;
	if (_type == MNISTReaderType::Train)
		label_file += "/train-labels.idx1-ubyte";
	else
		label_file += "/t10k-labels.idx1-ubyte";
	LOG(INFO) << "Initializing labels from " << label_file;
	_ty.open(label_file, std::ios::binary);
	LOG_IF(FATAL, !_ty.is_open()) << "TY not open.";
	_ty.read((char*)&msb, sizeof(__int32));
	_ty.read((char*)&n, sizeof(__int32));
	n = ReverseInt(n);
	LOG_IF(FATAL, n != 60000 && n != 10000) << "TY error.";
	_ty_start_pos = _ty.tellg();
	_labels_buf = new float[_batch_size * 10];	
	_outputs[1]->initValue({ _batch_size, 10, 1, 1});	
	_init_backward = true;
	nextBatch();
}

void MNISTReader::initBackward() {
	_outputs[0]->initDiff();
	_outputs[1]->initDiff();
}

void MNISTReader::nextBatch() {
	if (_last_batch == true)
	{
		_tx.clear(); // Reached EOF. Must reset the flag.
		_tx.seekg(_tx_start_pos);
		_ty.clear();
		_ty.seekg(_ty_start_pos);
		_current_batch = 0;
	}
	else
	{
		_current_batch++;
	}
	size_t _remaining_samples = _num_total_samples - _current_batch*_batch_size;
	if (_remaining_samples > _batch_size)
	{
		_num_batch_samples = _batch_size;
		_last_batch = false;
	}
	else
	{
		_num_batch_samples = _remaining_samples;
		_last_batch = true;
	}

	for (int i = 0; i < _num_batch_samples * 10; i++)
		_labels_buf[i] = 0.0f;	
	for (int i = 0; i < _num_batch_samples; ++i)
	{
		_tx.read((char*)_temp_d, 28 * 28);
		for (int j = 0; j < 28 * 28; ++j)			
			_data_buf[i * 28 * 28 + j] = ((float)_temp_d[j] / 255.0f - 0.5f) * 2;		
		_ty.read((char*)&_temp_l, sizeof(_temp_l));
		_labels_buf[i * 10 + _temp_l] = 1.0f;
	}
	if (_init_forward)
		LOG_IF(FATAL, cudaMemcpy(_outputs[0]->value()->mutableData(), _data_buf, _outputs[0]->value()->sizeInBytes(), cudaMemcpyHostToDevice) != 0) << "cudaMemcpy [FAILED]";
	if (_init_backward)
		LOG_IF(FATAL, cudaMemcpy(_outputs[1]->value()->mutableData(), _labels_buf, _outputs[1]->value()->sizeInBytes(), cudaMemcpyHostToDevice) != 0) << "cudaMemcpy [FAILED]";
}

void MNISTReader::deinit() {
	if (_temp_d) delete[] _temp_d;
	if (_data_buf) delete[] _data_buf;
	if (_labels_buf) delete[] _labels_buf;
	_tx.close();
	_ty.close();
}

bool MNISTReader::isLastBatch() {
	return _last_batch;
}