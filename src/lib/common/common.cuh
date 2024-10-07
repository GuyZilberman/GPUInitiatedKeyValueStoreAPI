#ifndef COMMON_H
#define COMMON_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <cstdlib>

#define CUDA_ERRCHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); }

// Macro to stringify the expression
#define STRINGIFY(x) #x

// Macro for checking conditions in C++ code and printing them as a string if they fail.
#define ERRCHECK(condition) { cppAssert((condition), STRINGIFY(condition), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "cudaAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


inline void cppAssert(bool condition, const char *conditionStr, const char *file, int line, bool abort = true)
{
    if (!condition)
    {
        std::cerr << "Assertion failed: " << conditionStr << " in " << file << " at line " << line << std::endl;
        if (abort) std::exit(EXIT_FAILURE);
    }
}


#endif // !COMMON_H

