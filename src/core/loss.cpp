#include "core/loss.h"

Loss::Loss(const NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_loss_param() == false) << "param.has_loss_param() == false";
}