#pragma once
#include "Core.h"
#include "FileParsing.h"

class Engine {
public:
	static Engine& Get() {
		static Engine instance;
		return instance;
	}

	bool Initialize(
		WINDOW_DESC& window_desc,
		RENDERER_DESC& renderer_desc);
	int Run();
	bool EditStage(Stage* stage);
	void Shutdown();

	Renderer* GetRenderer();
private:
	Engine() :
		m_pWindow(nullptr),
		m_pCore(nullptr),
		m_pRenderer(nullptr),
		m_pStage(nullptr),
		m_isRunning(false)
	{}
	Window* m_pWindow;
	Core* m_pCore;
	
	Renderer* m_pRenderer;
	Stage* m_pStage;

	bool m_isRunning;
};