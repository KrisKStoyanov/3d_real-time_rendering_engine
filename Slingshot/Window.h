#pragma once
#include "D3D11GraphicsContext.h"

struct WINDOW_DESC {
	GraphicsContextType graphicsContextType;
	HINSTANCE hInstance;
	PCWSTR lpWindowName;
	DWORD dwStyle;
	DWORD dwExStyle;
	PCWSTR lpClassName;
	int xCoord, yCoord;
	int nWidth, nHeight;
	HWND hWndParent;
	HMENU hMenu;
	int nCmdShow;

	WINDOW_DESC(
		GraphicsContextType _graphicsContextType,
		HINSTANCE _hInstance,
		PCWSTR _lpWindowName,
		DWORD _dwStyle,
		int _nCmdShow = 5,
		DWORD _dwExStyle = 0,
		PCWSTR _lpClassName = L"SlingshotWindow",
		int _xCoord = CW_USEDEFAULT,
		int _yCoord = CW_USEDEFAULT,
		int _nWidth = CW_USEDEFAULT,
		int _nHeight = CW_USEDEFAULT,
		HWND _hWndParent = 0,
		HMENU _hMenu = 0) :
		graphicsContextType(_graphicsContextType),
		hInstance(_hInstance),
		lpWindowName(_lpWindowName),
		dwStyle(_dwStyle),
		nCmdShow(_nCmdShow),
		dwExStyle(_dwExStyle),
		lpClassName(_lpClassName),
		xCoord(_xCoord), yCoord(_yCoord),
		nWidth(_nWidth), nHeight(_nHeight),
		hWndParent(_hWndParent),
		hMenu(_hMenu) {}
};

class Window
{
public:
	Window() : m_hWnd(nullptr), m_pDesc(nullptr) {}
	BOOL Create(WINDOW_DESC * window_desc);
	int OnUpdate(bool & isRunning);
	GraphicsContext* GetGraphicsContext();
	BOOL Shutdown();
private:
	HWND m_hWnd;
	WINDOW_DESC* m_pDesc;
};

