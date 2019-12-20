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

class GraphicsContext
{
public:
	virtual LRESULT HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam) { return LRESULT(); }

	virtual void OnCreate(HWND hwnd) {};
	virtual void OnDestroy() {};
	virtual void OnPaint() {};
	virtual void OnResize() {};
	virtual void OnLButtonDown(int pixelX, int pixelY, DWORD flags) {};
	virtual void OnLButtonUp() {};
	virtual void OnMouseMove(int pixelX, int pixelY, DWORD flags) {};
};

