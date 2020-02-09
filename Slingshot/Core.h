#pragma once
#include "Window.h"
#include "Camera.h"

class Core
{
public:
	Core() : 
		m_pWindow(nullptr), 
		m_pGC(nullptr), 
		m_pCamera(nullptr),
		m_isRunning(false)
	{};
	bool Initialize(WINDOW_DESC * window_desc);
	int Run();
	void Shutdown();
private:
	Window* m_pWindow;
	GraphicsContext* m_pGC;
	Camera* m_pCamera;
	bool m_isRunning;
};

