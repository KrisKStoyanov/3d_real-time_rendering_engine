#pragma once
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include <Windows.h>
#include <shellapi.h>

#include <wrl.h>
#include <algorithm>
#include <cassert>
#include <chrono>

#include <windef.h>

#include "Helpers.h"
#include "D2DGraphicsContext.h"
#include "D3D11GraphicsContext.h"

enum class GC {
	D2D = 0,
	D3D11,
	D3D12
};

class Window
{
public:
	static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
	{
		GraphicsContext* gc;

		if (uMsg == WM_CREATE) {
			CREATESTRUCT* pCreate = reinterpret_cast<CREATESTRUCT*>(lParam);
			gc = reinterpret_cast<GraphicsContext*>(pCreate->lpCreateParams);
			SetWindowLongPtrW(hwnd, GWLP_USERDATA, (LONG_PTR)gc);
			gc->OnCreate(hwnd);
		}
		else {
			gc = (GraphicsContext*)GetWindowLongPtrW(hwnd, GWLP_USERDATA);
		}

		if (gc) {
			return gc->HandleMessage(uMsg, wParam, lParam);
		}
		else {
			return DefWindowProcW(hwnd, uMsg, wParam, lParam);
		}
	}

	Window(GC gc);
	~Window();

	BOOL Create(
		PCWSTR lpWindowName,
		DWORD dwStyle,
		DWORD dwExStyle = 0,
		int xCoord = CW_USEDEFAULT,
		int yCoord = CW_USEDEFAULT,
		int nWidth = CW_USEDEFAULT,
		int nHeight = CW_USEDEFAULT,
		HWND hWndParent = 0,
		HMENU hMenu = 0);

	void OnUpdate();
	void Show(int cmdShow);
private:
	GraphicsContext* m_GC = NULL;
	GraphicsContext* GetWindowGC(HWND hwnd);

	//Window Handle
	HWND m_hWnd;
	//OS Window Class Template Name
	const PCWSTR m_wcName = L"SlingshotWindow";
};

