#pragma once

#include <Windows.h>
#include <stdio.h>
#include <fcntl.h>
#include <io.h>
#include <iostream>
#include <fstream>
#include <thread>
#include <string>

//Pending C++20 Stable Release Refactor
//#include <concepts>

//template<class From, class To>
//concept convertible_to = std::is_convertible<From, To> &&
//requires(From(&f)()) {
//	static_cast<To>(f());
//};
void StreamOutputToConsole(const char* output, const unsigned int conSleepMs = 1000, FILE* stdStream = stdout);
