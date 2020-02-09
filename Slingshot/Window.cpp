#include "Window.h"

LRESULT CALLBACK UpdateProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	Core* pCore =
		reinterpret_cast<Core*>
		(GetWindowLongPtrW(hwnd, GWLP_USERDATA));

	if (pCore) {
		return pCore->HandleMessage(hwnd, uMsg, wParam, lParam);
	}
	else{
		DestroyWindow(hwnd);
		PostQuitMessage(0);
	}
	return DefWindowProcW(hwnd, uMsg, wParam, lParam);
}

LRESULT CALLBACK SetupProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if (uMsg == WM_CREATE) {
		CREATESTRUCT* pCreate = reinterpret_cast<CREATESTRUCT*>(lParam);
		CORE_DESC* pCoreDesc =
			reinterpret_cast<CORE_DESC*>(pCreate->lpCreateParams);

		Core* pCore = Core::Create(pCoreDesc, hWnd);
		if (pCore) {
			if (pCore->Initialize()) {
				SetWindowLongPtrW(
					hWnd, GWLP_USERDATA,
					reinterpret_cast<LONG_PTR>(pCore));
				SetWindowLongPtrW(
					hWnd, GWLP_WNDPROC,
					reinterpret_cast<LONG_PTR>(UpdateProc));
			}
		}			
	}
	return DefWindowProcW(hWnd, uMsg, wParam, lParam);
}

Window* Window::Create(WINDOW_DESC* window_desc, CORE_DESC* core_desc)
{
	return new Window(window_desc, core_desc);
}

Window::Window(
	WINDOW_DESC * window_desc,
	CORE_DESC * core_desc) : m_hWnd(nullptr), m_pDesc(nullptr)
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
		&core_desc);
	
	if (!m_hWnd) {
		return;
	}

	ShowWindow(m_hWnd, window_desc->nCmdShow);
	m_pDesc = window_desc;
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

Core* Window::GetMessageHandler()
{
	LONG_PTR ptr = GetWindowLongPtrW(m_hWnd, GWLP_USERDATA);
	Core* pCore = reinterpret_cast<Core*>(ptr);
	return pCore;
}

BOOL Window::Shutdown()
{
	DestroyWindow(m_hWnd);
	return UnregisterClassW(m_pDesc->lpClassName, m_pDesc->hInstance);
}
