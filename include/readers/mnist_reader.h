#pragma once

#include "core/reader.h"

#include <string>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>

#include "cuda.h"
#include "cudnn.h"

enum DeepFlowDllExport MNISTReaderType {
	Train,
	Test
};

class DeepFlowDllExport MNISTReader : public Reader {
public:
	MNISTReader(NodeParam param);
	int minNumInputs() { return 0; }
	int minNumOutputs() { return 2; }
	void nextBatch();
	void forward();
	void backward();
	void initForward();
	void initBackward();
	void deinit();
private:
	MNISTReaderType _type;
	std::string _folder_path;
	std::ifstream _tx;
	std::ifstream _ty;
	int _current_batch;
	std::streamoff _tx_start_pos, _ty_start_pos;
	size_t _num_total_samples;
	size_t _num_batch_samples;
	bool _last_batch;
	int _batch_size;
	float *_data_buf;
	float *_labels_buf;
	unsigned char *_temp_d;
	unsigned char _temp_l;
	bool _init_forward;
	bool _init_backward;
};