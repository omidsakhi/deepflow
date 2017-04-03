#pragma once

#include "core/node_observer.h"

#include <memory>
#include <list>

class VariableFinder : public NodeObserver {
public:
	VariableFinder(std::list < Node* > *list);
	void apply(Node*node);
protected:
	std::list < Node* > *_list;
};