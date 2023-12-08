#pragma once

#include <iostream>

//#define DEBUG // if defined, print pretty benchmark messages to std::cout
#define DUMP // if defined, print CSV style results to std::cout

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

// Taken from : https://codereview.stackexchange.com/questions/60627/add-color-to-terminal-output

#define BLACK_TEXT(x) "\033[30;1m" x "\033[0m"
#define RED_TEXT(x) "\033[31;1m" x "\033[0m"
#define GREEN_TEXT(x) "\033[32;1m" x "\033[0m"
#define YELLOW_TEXT(x) "\033[33;1m" x "\033[0m"
#define BLUE_TEXT(x) "\033[34;1m" x "\033[0m"
#define MAGENTA_TEXT(x) "\033[35;1m" x "\033[0m"
#define CYAN_TEXT(x) "\033[36;1m" x "\033[0m"
#define WHITE_TEXT(x) "\033[37;1m" x "\033[0m"

#define BOLD_BLACK_TEXT(x) "\033[1m\033[30m;1m" x "\033[0m"
#define BOLD_RED_TEXT(x) "\033[1m\033[31m;1m" x "\033[0m"
#define BOLD_GREEN_TEXT(x) "\033[1m\033[32m;1m" x "\033[0m"
#define BOLD_YELLOW_TEXT(x) "\033[1m\033[33m;1m" x "\033[0m"
#define BOLD_BLUE_TEXT(x) "\033[1m\033[34m;1m" x "\033[0m"
#define BOLD_MAGENTA_TEXT(x) "\033[1m\033[35m;1m" x "\033[0m"
#define BOLD_CYAN_TEXT(x) "\033[1m\033[36m;1m" x "\033[0m"
#define BOLD_WHITE_TEXT(x) "\033[1m\033[37m;1m" x "\033[0m"

 
