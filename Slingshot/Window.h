#pragma once
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <shellapi.h>

#include <windef.h>
#include <windowsx.h>

#include <wrl.h>
#include <wrl/client.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <vector>
#include <fstream>

struct WINDOW_DESC 
{
	HINSTANCE hInstance;
	PCWSTR lpWindowName = L"Slingshot Graphics";
	int nCmdShow = 5;
	int nWidth = 1280, nHeight = 720;
	DWORD dwStyle = WS_OVERLAPPEDWINDOW;
	DWORD dwExStyle = 0;
	PCWSTR lpClassName = L"SlingshotWindow";
	int xCoord = CW_USEDEFAULT, yCoord = CW_USEDEFAULT;
	HWND hWndParent = 0;
	HMENU hMenu = 0;
};

class Window
{
public:
	static Window* Create(WINDOW_DESC& window_desc);
	HWND GetHandle();
	void Shutdown();
private:
	Window(WINDOW_DESC& window_desc);
	HWND m_hWnd;
	WINDOW_DESC m_pDesc;
};

