#pragma once

#include <iostream>

#define DEBUG // if defined, print pretty benchmark messages to std::cout
// #define DUMP // if defined, print CSV style results to std::cout

#ifdef DEBUG
    #define DEBUG_OUT(x) do { std::cout << x; } while (0)
    #define DEBUG_ERR(x) do { std::cerr << x; } while (0)
#else 
    #define DEBUG_ERR(x)
    #define DEBUG_OUT(x)
#endif

#ifdef DUMP
    #define FILE_DUMP(x) do { std::cout << x; } while (0)
#else 
    #define FILE_DUMP(x)
#endif


 