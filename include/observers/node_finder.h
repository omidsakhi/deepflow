#pragma once

#include "core/node_observer.h"

#include <memory>
#include <list>

template<typename T>
class NodeFinder : public NodeObserver {
public:
	NodeFinder(std::list<std::shared_ptr<T>> &list);
	void apply(std::shared_ptr<Node> node);
protected:
	std::list<std::shared_ptr<T>> &_list;
};

template<typename T>
NodeFinder<T>::NodeFinder(std::list<std::shared_ptr<T>> &list) : _list(list) {
	list.clear();
}

template<typename T>
void NodeFinder<T>::apply(std::shared_ptr<Node> node) {
	if (dynamic_cast<T*>(node.get()) != NULL)	
		_list.push_back(std::dynamic_pointer_cast<T>(node));
}