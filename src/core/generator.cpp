#include "core/generator.h"

Generator::Generator(const deepflow::NodeParam &_block_param) {
	LOG_IF(FATAL, _block_param.has_generator_param() == false) << "param.has_generator_param() == false";
}
