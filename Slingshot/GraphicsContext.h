#pragma once
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include <Windows.h>
#include <shellapi.h>

#include <windef.h>
#include <windowsx.h>

//D2D
#include <d2d1.h>

//STD
#include <wrl.h>
#include <algorithm>
#include <cassert>
#include <chrono>

class GraphicsContext
{
public:
	GraphicsContext();
	~GraphicsContext();

	virtual LRESULT HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam);

	virtual void OnCreate(HWND hwnd);
	virtual void OnDestroy();
	virtual void OnPaint();
	virtual void OnResize();
	virtual void OnLButtonDown(int pixelX, int pixelY, DWORD flags);
	virtual void OnLButtonUp();
	virtual void OnMouseMove(int pixelX, int pixelY, DWORD flags);
};

