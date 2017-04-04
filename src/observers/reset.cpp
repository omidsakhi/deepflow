#include "observers/reset.h"

void ResetObserver::apply(std::shared_ptr<Node> node) {
	node->setVisited(false);
}