#pragma once
#include "Renderer.h"
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

	LRESULT CALLBACK HandleWindowMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
	Renderer* GetRenderer();
private:
	Engine() :
		m_pWindow(nullptr),
		m_pRenderer(nullptr),
		m_pStage(nullptr),
		m_isRunning(false)
	{}
	Window* m_pWindow;
	
	Renderer* m_pRenderer;
	Stage* m_pStage;

	bool m_isRunning;
};