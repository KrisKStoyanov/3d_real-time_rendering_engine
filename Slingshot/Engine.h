#pragma once
#include "Core.h"
#include "FileParsing.h"

class Engine {
public:
	Engine() :
		m_pWindow(nullptr),
		m_pCore(nullptr),
		m_pStage(nullptr),
		m_isRunning(false)
	{}
	bool Initialize(
		WINDOW_DESC* window_desc,
		RENDERER_DESC* renderer_desc);
	int Run();
	bool SetupStage(Stage* stage);
	unsigned int GetActiveStageID();
	void Shutdown();
private:
	Window* m_pWindow;
	Core* m_pCore;
	
	Stage* m_pStage;

	bool m_isRunning;
};