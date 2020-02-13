#pragma once
#include "Stage.h"

class Core
{
public:
	static Core* Create(HWND hWnd);
	LRESULT CALLBACK HandleMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
	bool InitializeRenderer(RENDERER_DESC* core_desc);
	bool OnUpdate(Stage* stage);
	void Shutdown(void);

	Renderer* GetRenderer();
private:
	Core(HWND hWnd);
	bool m_isActive;
	HWND m_hWnd;

	Renderer* m_pRenderer;
};

