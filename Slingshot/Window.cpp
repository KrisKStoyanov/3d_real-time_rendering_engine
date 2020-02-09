#include "Window.h"

LRESULT CALLBACK UpdateProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	GraphicsContext* pGC =
		reinterpret_cast<GraphicsContext*>
		(GetWindowLongPtrW(hwnd, GWLP_USERDATA));

	if (pGC) {
		return pGC->HandleMessage(uMsg, wParam, lParam);
	}
	else{
		DestroyWindow(hwnd);
		PostQuitMessage(0);
	}
	return DefWindowProcW(hwnd, uMsg, wParam, lParam);
}

LRESULT CALLBACK SetupProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if (uMsg == WM_CREATE) {
		GraphicsContext* pGC;
		CREATESTRUCT* pCreate = reinterpret_cast<CREATESTRUCT*>(lParam);
		GraphicsContextType* gcTag = reinterpret_cast<GraphicsContextType*>(pCreate->lpCreateParams);
		switch (*gcTag) {
		case GraphicsContextType::D3D11:
			pGC = new D3D11GraphicsContext(hwnd);
			break;
		default:
			pGC = new D3D11GraphicsContext(hwnd);
			break;
		}

		SetWindowLongPtrW(
			hwnd, GWLP_USERDATA,
			reinterpret_cast<LONG_PTR>(pGC));
		SetWindowLongPtrW(
			hwnd, GWLP_WNDPROC,
			reinterpret_cast<LONG_PTR>(UpdateProc));
		pGC->OnCreate(hwnd);
	}
	return DefWindowProcW(hwnd, uMsg, wParam, lParam);
}

BOOL Window::Create(
	WINDOW_DESC * window_desc)
{

	WNDCLASSEXW wc = {};
	wc.cbSize = sizeof(WNDCLASSEX);
	wc.style = window_desc->dwExStyle;
	wc.lpfnWndProc = SetupProc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = window_desc->hInstance;
	wc.hIcon = NULL;
	wc.hCursor = LoadCursorW(NULL, IDC_ARROW);
	wc.hbrBackground = NULL;
	wc.lpszMenuName = NULL;
	wc.lpszClassName = window_desc->lpClassName;
	wc.hIconSm = NULL;

	if (!RegisterClassExW(&wc)) {
		return false;
	};

	m_hWnd = CreateWindowExW(
		window_desc->dwExStyle,
		window_desc->lpClassName,
		window_desc->lpWindowName,
		window_desc->dwStyle,
		window_desc->xCoord,
		window_desc->yCoord,
		window_desc->nWidth,
		window_desc->nHeight,
		window_desc->hWndParent,
		window_desc->hMenu,
		window_desc->hInstance,
		&window_desc->graphicsContextType);
	
	if (!m_hWnd) {
		return false;
	}

	ShowWindow(m_hWnd, window_desc->nCmdShow);
	m_pDesc = window_desc;

	return true;
}

int Window::OnUpdate(bool & isRunning)
{
	MSG msg = {};
	while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
		TranslateMessage(&msg);
		DispatchMessageW(&msg);
		if (msg.message == WM_QUIT) {
			isRunning = false;
		}
	}
	return (int)msg.wParam;
}

GraphicsContext* Window::GetGraphicsContext()
{
	LONG_PTR ptr = GetWindowLongPtrW(m_hWnd, GWLP_USERDATA);
	GraphicsContext* pGC = reinterpret_cast<GraphicsContext*>(ptr);
	return pGC;
}

BOOL Window::Shutdown()
{
	DestroyWindow(m_hWnd);
	return UnregisterClassW(m_pDesc->lpClassName, m_pDesc->hInstance);
}
