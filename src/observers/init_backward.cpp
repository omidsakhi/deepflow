#include "observers/init_backward.h"
#include "core/node.h"

void InitBackwardObserver::apply(Node *node) {
	node->initBackward();
}