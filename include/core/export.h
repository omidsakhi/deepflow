#pragma once

#ifdef DEEPFLOW_DLL_EXPORT
#define DeepFlowDllExport __declspec( dllexport )
#elif DEEPFLOW_DLL_IMPORT
#define DeepFlowDllExport __declspec( dllimport )
#else
#define DeepFlowDllExport 
#endif // DEEPFLOW_DLL_LIB


