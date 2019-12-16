#include "CUDAGraphicsContext.h"

CUDAGraphicsContext::CUDAGraphicsContext()
{
}

CUDAGraphicsContext::~CUDAGraphicsContext()
{
}

LRESULT CUDAGraphicsContext::HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg) {
	case WM_SIZE:
	{
		OnResize();
	}
	break;
	case WM_PAINT:
	{
		OnPaint();
	}
	break;
	case WM_LBUTTONDOWN:
	{
		OnLButtonDown(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
	}
	break;
	case WM_LBUTTONUP:
	{
		OnLButtonUp();
	}
	break;
	case WM_MOUSEMOVE:
	{
		OnMouseMove(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
	}
	break;
	case WM_SYSKEYDOWN:
	case WM_KEYDOWN:
	{
		bool alt = (::GetAsyncKeyState(VK_MENU) & 0x8000) != 0;

		switch (wParam) {
		case 'C':
		{
			if (alt) {
			
			}
		}
		break;
		case VK_ESCAPE:
		{
			OnDestroy();
			DestroyWindow(m_hWnd);
		}
		}
		return 0;
	}
	break;
	case WM_SYSCHAR:
		break;
		//case WM_SETCURSOR:
		//	if (LOWORD(lParam) == HTCLIENT) {
		//		SetCursor(m_hCursor);
		//		return TRUE;
		//	}
		//	break;
	case WM_DESTROY:
	{
		PostQuitMessage(0);
		return 0;
	}
	break;
	default:
	{
		DefWindowProcW(m_hWnd, uMsg, wParam, lParam);
	}
	break;
	}
}

void CUDAGraphicsContext::OnCreate(HWND hwnd)
{
	m_hWnd = hwnd;
	HC::ScheduleRenderKernel(1280, 720);
}

void CUDAGraphicsContext::OnDestroy()
{
}

void CUDAGraphicsContext::OnPaint()
{

}

void CUDAGraphicsContext::OnResize()
{
}

void CUDAGraphicsContext::OnLButtonDown(int pixelX, int pixelY, DWORD flags)
{
}

void CUDAGraphicsContext::OnLButtonUp()
{
}

void CUDAGraphicsContext::OnMouseMove(int pixelX, int pixelY, DWORD flags)
{
}
