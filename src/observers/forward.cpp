#include "observers/forward.h"
#include "core/node.h"

void ForwardObserver::apply(Node *node) {	
	node->forward();
}