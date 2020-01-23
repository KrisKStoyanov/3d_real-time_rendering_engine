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
		GC* gcTag = reinterpret_cast<GC*>(pCreate->lpCreateParams);
		switch (*gcTag) {
		case GC::D3D11:
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
	GC graphicsContext,
	HINSTANCE hInstance,
	PCWSTR lpWindowName, 
	DWORD dwStyle, 
	DWORD dwExStyle, 
	int xCoord, int yCoord, 
	int nWidth, int nHeight, 
	HWND hWndParent, HMENU hMenu,
	int nCmdShow)
{

	WNDCLASSEXW wc = {};
	wc.cbSize = sizeof(WNDCLASSEX);
	wc.style = dwExStyle;
	wc.lpfnWndProc = SetupProc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = hInstance;
	wc.hIcon = NULL;
	wc.hCursor = LoadCursorW(NULL, IDC_ARROW);
	wc.hbrBackground = NULL;
	wc.lpszMenuName = NULL;
	wc.lpszClassName = m_wcName;
	wc.hIconSm = NULL;

	if (!RegisterClassExW(&wc)) {
		return -1;
	};

	m_hWnd = CreateWindowExW(
		dwExStyle,
		m_wcName,
		lpWindowName,
		dwStyle,
		xCoord,
		yCoord,
		nWidth,
		nHeight,
		hWndParent,
		hMenu,
		hInstance,
		&graphicsContext);
	
	ShowWindow(m_hWnd, nCmdShow);

	return (m_hWnd ? TRUE : FALSE);
}

void Window::OnUpdate()
{
	MSG msg = {};
	while (GetMessageW(&msg, NULL, 0, 0)) {
		TranslateMessage(&msg);
		DispatchMessageW(&msg);
	}
}

GraphicsContext* Window::GetWindowGC(HWND hwnd)
{
	LONG_PTR ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA);
	GraphicsContext* pGC = reinterpret_cast<GraphicsContext*>(ptr);
	return pGC;
}