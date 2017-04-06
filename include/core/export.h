#pragma once

#ifdef DEEPFLOW_DLL_EXPORT
#	define DeepFlowDllExport __declspec( dllexport )
#	define EXPIMP_TEMPLATE
#elif DEEPFLOW_DLL_IMPORT
#	define DeepFlowDllExport __declspec( dllimport )
#	define EXPIMP_TEMPLATE extern
#else
#	define DeepFlowDllExport
#	define EXPIMP_TEMPLATE
#endif // DEEPFLOW_DLL_LIB


