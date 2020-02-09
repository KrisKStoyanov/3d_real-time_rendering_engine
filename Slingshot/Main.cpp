#include "Macros.h"
#include "Core.h"

//SAL annotations for entry point parameter config 
//(https://docs.microsoft.com/en-us/visualstudio/code-quality/understanding-sal?view=vs-2015)
int CALLBACK wWinMain(
	_In_ HINSTANCE hInstance, 
	_In_opt_ HINSTANCE hPrevInstance, 
	_In_ PWSTR lpCmdLine, 
	_In_ int nCmdShow) 
{
	int status = EXIT_FAILURE;

	Core* core = new Core();
	if (core->Initialize(
			&WINDOW_DESC(GraphicsContextType::D3D11,
			hInstance,
			L"Slingshot Graphics",
			WS_OVERLAPPEDWINDOW,
			nCmdShow))) {
		status = core->Run();
	}
	SAFE_SHUTDOWN(core);

	return status;
}