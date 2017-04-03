#include "observers/init_forward.h"
#include "core/node.h"

void InitForwardObserver::apply(Node *node) {
	node->initForward();
}