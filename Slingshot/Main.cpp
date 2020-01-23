#include "Renderer.h"

//SAL annotations for entry point parameter config 
//(https://docs.microsoft.com/en-us/visualstudio/code-quality/understanding-sal?view=vs-2015)
int CALLBACK wWinMain(
	_In_ HINSTANCE hInstance, 
	_In_opt_ HINSTANCE hPrevInstance, 
	_In_ PWSTR lpCmdLine, 
	_In_ int nCmdShow) 
{
	Window win;
	if (win.Create(GC::D3D11, 
		hInstance,
		L"Slingshot Graphics", 
		WS_OVERLAPPEDWINDOW, 
		nCmdShow)) {
		Renderer renderer;
		if (renderer.OnStart(&win)) {
			renderer.OnUpdate();
		}
	}
	return 0;
}