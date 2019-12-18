#include "Window.h"

Window::Window(GC gc)
{
	switch (gc) {
	case GC::D3D11:
	{
		m_GC = new D3D11GraphicsContext();
	}
	break;
	case GC::D3D12:
	{
		m_GC = new D3D12GraphicsContext();
	}
	break;
	default:
	{
		m_GC = new D3D11GraphicsContext();
	}
	break;
	}
}

Window::~Window()
{
}

BOOL Window::Create(PCWSTR lpWindowName, DWORD dwStyle, DWORD dwExStyle, int xCoord, int yCoord, int nWidth, int nHeight, HWND hWndParent, HMENU hMenu)
{
	//Behaviour template for extended window class template (supporting Unicode (W) chars)
	WNDCLASSEXW wc = {};

	//Obtain application instance signature
	HINSTANCE hInstance = GetModuleHandleW(NULL);

	wc.cbSize = sizeof(WNDCLASSEX);
	wc.style = dwExStyle;
	wc.lpfnWndProc = &WindowProc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = hInstance;
	wc.hIcon = NULL;
	wc.hCursor = LoadCursorW(NULL, IDC_ARROW);
	wc.hbrBackground = NULL;
	wc.lpszMenuName = NULL;
	wc.lpszClassName = m_wcName;
	wc.hIconSm = NULL;

	RegisterClassExW(&wc);

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
