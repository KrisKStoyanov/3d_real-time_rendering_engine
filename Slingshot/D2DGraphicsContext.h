#pragma once
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include <Windows.h>
#include <shellapi.h>

#include <windef.h>
#include <windowsx.h>

//D2D
#include <d2d1.h>

//STD
#include <wrl.h>
#include <algorithm>
#include <cassert>
#include <chrono>

#include "GraphicsContext.h"

enum class UserMode {
	Draw,
	Select,
	Drag
};

class D2DGraphicsContext : public GraphicsContext
{
public:
	D2DGraphicsContext();
	~D2DGraphicsContext();

	LRESULT HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam);

	//D2D
	ID2D1Factory* pFactory = NULL;
	ID2D1HwndRenderTarget* pRenderTarget = NULL;
	ID2D1SolidColorBrush* pBrush = NULL;

	D2D1_ELLIPSE ellipse;

	virtual void OnCreate(HWND hwnd);
	virtual void OnDestroy();
	virtual void OnPaint();
	virtual void OnResize();
	virtual void OnLButtonDown(int pixelX, int pixelY, DWORD flags);
	virtual void OnLButtonUp();
	virtual void OnMouseMove(int pixelX, int pixelY, DWORD flags);

	void CalculateLayout();
	HRESULT CreateGraphicsResources();
	void DiscardGraphicsResources();
	void CaptureCursor();
	void SetMode(UserMode mode);

	//DPI - Dots Per Inch (MS graphics measurement - 96 points Per Inch)
	//Alternative to DIP - Device-Independant Pixels D2D graphics metric
	void RetrieveDPIScale();
	D2D1_POINT_2F PixelsToDips(FLOAT xCoord, FLOAT yCoord);

	HWND m_hWnd;
	RECT m_ClientRect;

	float m_DPIScaleX = 1.0f;
	float m_DPIScaleY = 1.0f;
	D2D1_POINT_2F m_ptMouse = D2D1::Point2F();
	bool m_CaptureCursor = false;
	HCURSOR m_hCursor;

	UserMode m_Mode = UserMode::Draw;
};


