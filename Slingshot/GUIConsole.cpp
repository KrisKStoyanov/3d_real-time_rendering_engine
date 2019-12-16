#include "GUIConsole.h"

void StreamOutputToConsole(const char* output, const unsigned int conSleepMs, FILE* stdStream) {

	if (AllocConsole()) {

		if (stdStream == stdout) {
			
			FILE* pFstdout = stdout;
			freopen_s(&pFstdout, "CONOUT$", "w", stdout);
			std::cout << output << std::flush;
			fclose(stdout);
			std::this_thread::sleep_for(std::chrono::milliseconds(conSleepMs));
			
		}
		else if (stdStream == stderr) {

			FILE* pFstderr = stderr;
			freopen_s(&pFstderr, "CONOUT$", "w", stderr);
			std::cerr << output << std::flush;
			fclose(stderr);
			std::this_thread::sleep_for(std::chrono::milliseconds(conSleepMs));
		}
		
		FreeConsole();
	}
}
