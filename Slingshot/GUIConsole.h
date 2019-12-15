#pragma once

#include <Windows.h>
#include <stdio.h>
#include <fcntl.h>
#include <io.h>
#include <iostream>
#include <fstream>

void StreamToConsole() {
	static const WORD maxLines = 500;
	FILE* pFile;
	CONSOLE_SCREEN_BUFFER_INFO info;
	if (AllocConsole()) {
		HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
		if (GetConsoleScreenBufferInfo(h, &info)) {
			info.dwSize.Y = maxLines;
			SetConsoleScreenBufferSize(h, info.dwSize);

			unsigned long long lStdHandle = (unsigned long long)h;
			int hConHandle = _open_osfhandle(lStdHandle, _O_TEXT);
			
			pFile = _fdopen(hConHandle, "w");
			*stdin = *pFile;
			setvbuf(stdin, NULL, _IONBF, 0);
			std::ios::sync_with_stdio(true);
		}
	}
}
