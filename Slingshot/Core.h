#pragma once
#include "Renderer.h"

struct CORE_DESC {
	GraphicsContextType graphicsContextType;
	CORE_DESC(
		GraphicsContextType _graphicsContextType) :
		graphicsContextType(_graphicsContextType) {}
};

class Core
{
public:
	static Core* Create(CORE_DESC* core_desc, HWND hWnd);
	LRESULT CALLBACK HandleMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
	bool Initialize(HWND hWnd);
	void OnUpdate(void);
	void Shutdown(void);
private:
	Core(CORE_DESC* core_desc, HWND hWnd);
	CORE_DESC* m_pDesc;

	Renderer* m_pRenderer;
};

