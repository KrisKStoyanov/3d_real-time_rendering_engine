#pragma once

#include "D3D11GraphicsContext.h"

class Window
{
public:
	Window() : m_hWnd(NULL) {}

	BOOL Create(
		GC graphicsContext,
		HINSTANCE hInstance,
		PCWSTR lpWindowName,
		DWORD dwStyle,
		DWORD dwExStyle = 0,
		int xCoord = CW_USEDEFAULT,
		int yCoord = CW_USEDEFAULT,
		int nWidth = CW_USEDEFAULT,
		int nHeight = CW_USEDEFAULT,
		HWND hWndParent = 0,
		HMENU hMenu = 0,
		int nCmdShow = 5);

	void OnUpdate();
	GraphicsContext* GetGraphicsContext();
private:
	GC gc;
	HWND m_hWnd;
	const PCWSTR m_wcName = L"SlingshotWindow";
};

