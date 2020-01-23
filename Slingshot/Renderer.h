#pragma once
#include "Window.h"
#include "Camera.h"

class Renderer
{
public:
	Renderer() : m_Window(NULL), m_GC(NULL), m_Camera(NULL) {}

	BOOL OnStart(Window* win);
	void OnUpdate();
private:
	Window* m_Window;
	GraphicsContext* m_GC;
	Camera* m_Camera;
};

