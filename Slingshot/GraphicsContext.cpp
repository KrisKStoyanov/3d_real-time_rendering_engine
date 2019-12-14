#include "GraphicsContext.h"

GraphicsContext::GraphicsContext()
{
}

GraphicsContext::~GraphicsContext()
{
}

LRESULT GraphicsContext::HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam)
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
		CaptureCursor();
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
				m_CaptureCursor = !m_CaptureCursor;
				CaptureCursor();
			}
		}
		break;
		case VK_ESCAPE:
		{
			if (MessageBoxW(m_hWnd, L"Confirm Exit?", L"Slingshot Engine", MB_OKCANCEL) == IDOK) {
				OnDestroy();
				DestroyWindow(m_hWnd);
			}
		}
		}
		return 0;
	}
	break;
	case WM_SYSCHAR:
		break;
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

void GraphicsContext::OnCreate()
{
	if (SUCCEEDED(D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &pFactory))) {
		RetrieveDPIScale();
		m_CaptureCursor = true;
	}
}

void GraphicsContext::OnDestroy()
{
	DiscardGraphicsResources();
	pFactory->Release();
	pFactory = NULL;
}

void GraphicsContext::CalculateLayout()
{
	if (pRenderTarget != NULL) {
		D2D1_SIZE_F size = pRenderTarget->GetSize();
		const float xCoord = size.width / 2;
		const float yCoord = size.height / 2;
		const float radius = std::min(xCoord, yCoord);
		ellipse = D2D1::Ellipse(D2D1::Point2F(xCoord, yCoord), radius, radius);
	}
}

HRESULT GraphicsContext::CreateGraphicsResources()
{
	HRESULT hr = S_OK;
	if (pRenderTarget == NULL) {
		RECT rc;
		GetClientRect(m_hWnd, &rc);

		D2D1_SIZE_U size = D2D1::SizeU(rc.right, rc.bottom);

		hr = pFactory->CreateHwndRenderTarget(
			D2D1::RenderTargetProperties(),
			D2D1::HwndRenderTargetProperties(m_hWnd, size),
			&pRenderTarget);

		if (SUCCEEDED(hr)) {
			const D2D1_COLOR_F color = D2D1::ColorF(1.0f, 0.5f, 0.0f);
			hr = pRenderTarget->CreateSolidColorBrush(color, &pBrush);

			if (SUCCEEDED(hr)) {
				CalculateLayout();
			}
		}
	}
	return hr;
}

void GraphicsContext::DiscardGraphicsResources()
{
	pRenderTarget->Release();
	pRenderTarget = NULL;
	pBrush->Release();
	pBrush = NULL;
}

void GraphicsContext::CaptureCursor()
{
	if (m_CaptureCursor) {
		RECT rc;
		GetClientRect(m_hWnd, &rc);

		POINT pt = { rc.left, rc.top };
		POINT pt2 = { rc.right, rc.bottom };
		ClientToScreen(m_hWnd, &pt);
		ClientToScreen(m_hWnd, &pt2);

		SetRect(&rc, pt.x, pt.y, pt2.x, pt2.y);

		ClipCursor(&rc);
	}
	else {
		ClipCursor(NULL);
	}
}

void GraphicsContext::RetrieveDPIScale()
{
	HDC hdc = GetDC(m_hWnd);
	m_DPIScaleX = GetDeviceCaps(hdc, LOGPIXELSX) / 96.0f;
	m_DPIScaleY = GetDeviceCaps(hdc, LOGPIXELSY) / 96.0f;
	ReleaseDC(m_hWnd, hdc);
}

D2D1_POINT_2F GraphicsContext::PixelsToDips(FLOAT xCoord, FLOAT yCoord)
{
	return D2D1::Point2F(static_cast<float>(xCoord) / m_DPIScaleX, static_cast<float>(yCoord) / m_DPIScaleY);
}

void GraphicsContext::OnPaint()
{
	HRESULT hr = CreateGraphicsResources();
	if (SUCCEEDED(hr)) {
		PAINTSTRUCT ps;
		BeginPaint(m_hWnd, &ps);
		pRenderTarget->BeginDraw();
		pRenderTarget->Clear(D2D1::ColorF(D2D1::ColorF::SkyBlue));
		pRenderTarget->FillEllipse(ellipse, pBrush);

		hr = pRenderTarget->EndDraw();
		if (FAILED(hr) || hr == D2DERR_RECREATE_TARGET) {
			DiscardGraphicsResources();
			CreateGraphicsResources();
		}
		EndPaint(m_hWnd, &ps);
	}
}

void GraphicsContext::OnResize()
{
	if (pRenderTarget != NULL) {
		RECT rc;
		GetClientRect(m_hWnd, &rc);
		D2D1_SIZE_U size = D2D1::SizeU(rc.right, rc.bottom);
		pRenderTarget->Resize(size);
		CalculateLayout();
		InvalidateRect(m_hWnd, NULL, FALSE);
	}
}

void GraphicsContext::OnLButtonDown(int pixelX, int pixelY, DWORD flags)
{
	SetCapture(m_hWnd);
	ellipse.point = m_ptMouse = PixelsToDips(pixelX, pixelY);
	ellipse.radiusX = ellipse.radiusY = 1.0f;
	InvalidateRect(m_hWnd, NULL, FALSE);
}

void GraphicsContext::OnLButtonUp()
{
	ReleaseCapture();
}

void GraphicsContext::OnMouseMove(int pixelX, int pixelY, DWORD flags)
{
	if (flags & MK_LBUTTON) {
		const D2D1_POINT_2F dips = PixelsToDips(pixelX, pixelY);
		
		const float width = (dips.x - m_ptMouse.x) / 2;
		const float height = (dips.y - m_ptMouse.y) / 2;
		const float x1 = m_ptMouse.x + width;
		const float y1 = m_ptMouse.y + height;

		ellipse = D2D1::Ellipse(D2D1::Point2F(x1, y1), width, height);

		InvalidateRect(m_hWnd, NULL, FALSE);
	}
}
