#pragma once

#include "D3D11GraphicsContext.h"
#include "D3D12GraphicsContext.h"

enum class GC : UINT {
	D3D11 = 0,
	D3D12,
	OpenGL,
	Vulkan
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

