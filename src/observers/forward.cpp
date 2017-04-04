#include "observers/forward.h"
#include "core/node.h"

void ForwardObserver::apply(std::shared_ptr<Node> node) {
	node->forward();
}