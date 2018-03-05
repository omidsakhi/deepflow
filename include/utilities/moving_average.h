#ifndef MOVINGAVERAGE_H
#define MOVINGAVERAGE_H

#include "core\export.h"

class DeepFlowDllExport MovingAverage {
public:
	MovingAverage(int n) {
		_total = n;
		_data = new float[n];
		for (int i = 0; i < n; ++i)
			_data[i] = 0;
		_current = 0;
		_sum = 0;
		_ready = false;
	}
	~MovingAverage() {
		delete[]_data;
	}
	void add(float value) {
		_prev_sum = _sum;
		_sum -= _data[_current];
		_data[_current] = value;
		_sum += value;
		_current++;
		if (_current >= _total)
		{
			_ready = true;
			_current = 0;
		}
	}
	float result() {
		return _ready? _sum / _total : _sum / _current;
	}
	float previous() {
		return _prev_sum / _total;
	}
	bool is_ready() {
		return _ready;
	}
	int count() {
		return _total;
	}
protected:
	float *_data;
	float _sum;
	float _prev_sum;
	unsigned int _current;
	unsigned int _total;
	bool _ready;

};

#endif