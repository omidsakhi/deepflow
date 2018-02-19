#include "nodes/matmul.h"

#include <glog/logging.h>

MatMul::MatMul(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_matmul_param() == false) << "param.has_matmul_param() == false";
}

void MatMul::init() {	
	auto a = _inputs[0];
	auto b = _inputs[1];
	
	auto ad = a->value()->dims();
	auto bd = b->value()->dims();

	_row_A = ad[0];
	_col_A = ad[1] * ad[2] * ad[3];
	_row_B = bd[0];
	_col_B = bd[1] * bd[2] * bd[3];
	
	LOG_IF(FATAL, _col_A != _row_B) << "_col_A != _row_B - " << a->value()->shape() << " * " << b->value()->shape();
	
	_outputs[0]->initValue({ _row_A, bd[1], bd[2], bd[3] });

	LOG(INFO) << "InnerProduct " << _name << " - " << _outputs[0]->value()->shape();	
	
	cublasCreate(&_handle);	

	_outputs[0]->initDiff();

}

void MatMul::forward() {	
	auto a = _inputs[0];
	auto b = _inputs[1];
	auto c = _outputs[0];

	// C(row_A,col_B) = A(row_A,col_A) * B(row_B,col_B)
	LOG_IF(FATAL, cublasSgemm(_handle, CUBLAS_OP_N, CUBLAS_OP_N, _col_B, _row_A, _row_B, &one, (float *) b->value()->data(), _col_B, (float *) a->value()->data(), _col_A, &zero, (float*) c->value()->mutableData(), _col_B) != 0) << "cublasSgemm [FAILED]";	
}

void MatMul::backward() {			
	auto a = _inputs[0];
	auto b = _inputs[1];
	auto c = _outputs[0];

	if (_inputs[0]->connectedNode()) {
		// col_A = row_B
		//A(row_A,col_A) = diff(row_A,col_B) * B(row_B,col_B).T		
		LOG_IF(FATAL, cublasSgemm(_handle, CUBLAS_OP_T, CUBLAS_OP_N, _row_B, _row_A, _col_B, &one, (float*)b->value()->data(), _col_B, (float*)c->diff()->data(), _col_B, &zero, (float*)a->diff()->mutableData(), _col_A) != 0) << "[FAILED] in " << _name;
	}
	
	if (_inputs[1]->connectedNode()) {
		//B(row_B,col_B) = A(row_A,col_A).T * diff(row_A,col_B)		
		LOG_IF(FATAL, cublasSgemm(_handle, CUBLAS_OP_N, CUBLAS_OP_T, _col_B, _col_A, _row_A, &one, (float *)c->diff()->data(), _col_B, (float *)a->value()->data(), _col_A, &zero, (float*)b->diff()->mutableData(), _col_B) != 0) << "[FAILED] in " << _name;
	}
	
}

std::string MatMul::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.matmul(" + _input_name_for_cpp(0) + ", " + _input_name_for_cpp(1) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
