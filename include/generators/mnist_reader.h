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

class DeepFlowDllExport MNISTReader : public Generator, public Node {
public:
	enum MNISTReaderType {
		Train,
		Test
	};
	enum MNISTOutputType {
		Data,
		Labels
	};
	MNISTReader(const deepflow::NodeParam &_block_param);
	int minNumInputs() { return 0; }
	int minNumOutputs() { return 1; }
	void nextBatch();
	void initForward();
	void initBackward();
	void deinit();
	bool isLastBatch();	
	std::string to_cpp() const;
	ForwardType forwardType() { return ALWAYS_FORWARD; }
	BackwardType backwardType() { return NEVER_BACKWARD; }
private:
	std::string _file_path;
	MNISTReaderType _reader_type;
	MNISTOutputType _output_type;
	std::string _folder_path;
	std::ifstream _tx;	
	int _current_batch = 0;
	int _batch_size = 0;
	std::streamoff _tx_start_pos;
	size_t _num_total_samples;
	size_t _num_batch_samples;
	bool _last_batch = false;	
	float *_buf = NULL;	
	unsigned char *_temp;	
};