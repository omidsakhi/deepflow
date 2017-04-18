#pragma once

#include "core/generator.h"

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

class DeepFlowDllExport MNISTReader : public Generator, public Node {
public:
	MNISTReader(const NodeParam &param);	
	int minNumInputs() { return 0; }
	int minNumOutputs() { return 2; }
	void nextBatch();
	void initForward();
	void initBackward();
	void deinit();
	bool isLastBatch();	
private:
	MNISTReaderType _type;
	std::string _folder_path;
	std::ifstream _tx;
	std::ifstream _ty;
	int _current_batch = 0;
	std::streamoff _tx_start_pos, _ty_start_pos;
	size_t _num_total_samples;
	size_t _num_batch_samples;
	bool _last_batch = false;
	int _batch_size = 0;
	float *_data_buf = NULL;
	float *_labels_buf = NULL;
	unsigned char *_temp_d;
	unsigned char _temp_l;
	bool _init_forward;
	bool _init_backward;
};