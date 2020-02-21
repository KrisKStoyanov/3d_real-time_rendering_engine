#include "Engine.h"

//SAL annotations for entry point parameter config 
//(https://docs.microsoft.com/en-us/visualstudio/code-quality/understanding-sal?view=vs-2015)
int CALLBACK wWinMain(
	_In_ HINSTANCE hInstance, 
	_In_opt_ HINSTANCE hPrevInstance, 
	_In_ PWSTR lpCmdLine, 
	_In_ int nCmdShow) 
{
	int status = EXIT_FAILURE;

	WINDOW_DESC window_desc;
	window_desc.hInstance = hInstance;
	window_desc.nCmdShow = nCmdShow;
	window_desc.nWidth = 1280;
	window_desc.nHeight = 720;

	RENDERER_DESC renderer_desc;

	Engine engine = Engine::Get();

	if (engine.Initialize(window_desc, renderer_desc))
	{
		status = engine.Run();
	}
	engine.Shutdown();

	return status;
}