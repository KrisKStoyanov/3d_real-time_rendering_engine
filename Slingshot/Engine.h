#pragma once
#include "Core.h"

class Engine {
public:
	Engine() :
		m_pWindow(nullptr),
		m_pCore(nullptr),
		m_isRunning(false)
	{}
	bool Initialize(
		WINDOW_DESC* window_desc,
		CORE_DESC* core_desc);
	int Run();
	void Shutdown();
private:
	Window* m_pWindow;
	Core* m_pCore;
	bool m_isRunning;
};