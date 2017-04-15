#include "core/reader.h"

Reader::Reader(const NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_reader_param() == false) << "param.has_reader_param() == false";
}
