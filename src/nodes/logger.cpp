#include "nodes/logger.h"

#include <fstream>

Logger::Logger(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_logger_param() == false) << "param.has_logger_param() == false";
	auto loggingParam = _param->logger_param();
	_filePath = loggingParam.file_path();
	_num_inputs = loggingParam.num_inputs();
	_logging_type = loggingParam.logging_type();
}

int Logger::minNumInputs() {
	return _num_inputs;
}

void Logger::init() {
	auto loggingParam = _param->logger_param();
	_raw_message = loggingParam.message();
}

void Logger::forward() {
	if (_enabled == false)
		return;
	std::string message = _raw_message;
	size_t start_pos = message.find("%{}");
	if (start_pos != std::string::npos) {
		float a, b;
		if (_logging_type == deepflow::ActionType::DIFFS) {
			a = atof(_inputs[0]->diff()->toString().c_str());
			b = atof(_inputs[1]->diff()->toString().c_str());
		}
		else {
			a = atof(_inputs[0]->value()->toString().c_str());
			b = atof(_inputs[1]->value()->toString().c_str());
		}
		message.replace(start_pos, 3, std::to_string(a / b * 100.0f));
	}
	for (int i = 0; i < _inputs.size(); ++i) {
		size_t start_pos = message.find("{" + std::to_string(i) + "}");
		if (start_pos == std::string::npos)
			continue;
		if (_logging_type == deepflow::ActionType::DIFFS)
			message.replace(start_pos, 3, _inputs[i]->diff()->toString());
		else
			message.replace(start_pos, 3, _inputs[i]->value()->toString());
	}

	std::ofstream outfile;
	outfile.open(_filePath, std::ios_base::app);
	outfile << message;		
	outfile.close();
}

void Logger::backward() {

}

std::string Logger::to_cpp() const
{
	std::string inputs = _input_name_for_cpp(0);
	for (int i = 1; i < _num_inputs; ++i) {
		inputs += ", " + _input_name_for_cpp(i);
	}

	std::string cpp = "df.logger( {" + inputs + "} , ";
	cpp += "\"" + _filePath + "\", ";
	std::string escaped_raw_message = _raw_message;
	for (int i = 0; i < escaped_raw_message.size(); ++i)
		if (escaped_raw_message[i] == '\n') {
			escaped_raw_message[i] = 'n';
		}
	cpp += "\"" + escaped_raw_message + "\", ";
	if (_logging_type == deepflow::ActionType::DIFFS) {
		cpp += "Logger::DIFFS, ";
	}
	else {
		cpp += "Logger::VALUES, ";
	}
	cpp += "\"" + _name + "\");";	
	return cpp;
}

void Logger::setEnabled(bool state)
{
	_enabled = state;
}
