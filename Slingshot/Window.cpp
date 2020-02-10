#include "Window.h"

LRESULT CALLBACK DefProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	return DefWindowProcW(hWnd, uMsg, wParam, lParam);
}

Window* Window::Create(WINDOW_DESC* window_desc)
{
	return new Window(window_desc);
}

Window::Window(WINDOW_DESC * window_desc) : m_hWnd(nullptr), m_pDesc(nullptr)
{
	WNDCLASSEXW wc = {};
	wc.cbSize = sizeof(WNDCLASSEX);
	wc.style = window_desc->dwExStyle;
	wc.lpfnWndProc = DefProc;
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
		return;
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
		nullptr);
	
	if (!m_hWnd) {
		return;
	}

	ShowWindow(m_hWnd, window_desc->nCmdShow);
	m_pDesc = window_desc;
}

HWND Window::GetHandle()
{
	return m_hWnd;
}

BOOL Window::Shutdown()
{
	DestroyWindow(m_hWnd);
	return UnregisterClassW(m_pDesc->lpClassName, m_pDesc->hInstance);
}
