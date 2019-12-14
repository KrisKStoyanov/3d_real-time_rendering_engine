#include "Window.h"

Window::Window(GraphicsContext* gc)
{
	m_GC = gc;
}

Window::~Window()
{
}

BOOL Window::Create(PCWSTR lpWindowName, DWORD dwStyle, DWORD dwExStyle, int xCoord, int yCoord, int nWidth, int nHeight, HWND hWndParent, HMENU hMenu)
{
	//Behaviour template for extended window class template (supporting Unicode (W) chars)
	WNDCLASS wc = {};

	//Obtain application instance signature
	HINSTANCE hInstance = GetModuleHandleW(NULL);

	wc.lpfnWndProc = &WindowProc;
	wc.hInstance = hInstance;
	wc.lpszClassName = m_wcName;

	RegisterClassW(&wc);

	m_GC = new (std::nothrow) GraphicsContext;

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
		m_GC);

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

void Window::Show(int cmdShow)
{
	ShowWindow(m_hWnd, cmdShow);
}

GraphicsContext* Window::GetWindowGC(HWND hwnd)
{
	LONG_PTR ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA);
	GraphicsContext* wsh = reinterpret_cast<GraphicsContext*>(ptr);
	return wsh;
}
