#include "observers/backward.h"
#include "core/node.h"

#include <glog/logging.h>

void BackwardObserver::apply(Node *node) {		
	node->backward();	

}