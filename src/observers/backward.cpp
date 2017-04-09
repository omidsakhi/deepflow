#include "observers/backward.h"
#include "core/node.h"

#include <glog/logging.h>

void BackwardObserver::apply(std::shared_ptr<Node> node) {
	node->backward();	
}