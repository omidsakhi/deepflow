#include "core/loss.h"

Loss::Loss(const deepflow::NodeParam &_block_param) : Node(_block_param) {
	LOG_IF(FATAL, _block_param.has_loss_param() == false) << "param.has_loss_param() == false";
}