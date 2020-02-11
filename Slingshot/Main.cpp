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

	Engine* engine = new Engine();
	if (engine->Initialize(
			&WINDOW_DESC(
				hInstance,
				L"Slingshot Graphics",
				nCmdShow,
				1280, 720),
			&RENDERER_DESC(
				GraphicsContextType::D3D11
			)
		)) {
		status = engine->Run();
	}
	SAFE_SHUTDOWN(engine);

	return status;
}