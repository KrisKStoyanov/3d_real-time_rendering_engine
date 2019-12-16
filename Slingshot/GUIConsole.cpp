#include "GUIConsole.h"

void StreamOutputToConsole(const char* output, FILE* stdStream, const unsigned int conSleepMs) {

	if (AllocConsole()) {

		if (stdStream == stdout) {
			
			FILE* pFstdout = stdout;
			freopen_s(&pFstdout, "CONOUT$", "w", stdout);
			std::cout << output << std::endl;
			fclose(stdout);
			std::this_thread::sleep_for(std::chrono::milliseconds(conSleepMs));
			
		}
		else if (stdStream == stderr) {

			FILE* pFstderr = stderr;
			freopen_s(&pFstderr, "CONOUT$", "w", stderr);
			std::cerr << output << std::endl;
			fclose(stderr);
			std::this_thread::sleep_for(std::chrono::milliseconds(conSleepMs));
		}
		
		FreeConsole();
	}
}
