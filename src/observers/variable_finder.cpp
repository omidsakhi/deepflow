#include "observers/variable_finder.h"

#include "core/node.h"
#include "core/variable.h"

#include <glog/logging.h>

VariableFinder::VariableFinder(std::list <Node*> *list) : _list(list) {
	
}

void VariableFinder::apply(Node *node) {		
	if (dynamic_cast<Variable*>(node) != NULL)
		_list->push_back(node);	
}