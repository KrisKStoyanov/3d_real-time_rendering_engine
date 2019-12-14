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

class GraphicsContext
{
public:
	GraphicsContext();
	~GraphicsContext();

	LRESULT HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam);

	//D2D
	ID2D1Factory* pFactory = NULL;
	ID2D1HwndRenderTarget* pRenderTarget = NULL;
	ID2D1SolidColorBrush* pBrush = NULL;

	D2D1_ELLIPSE ellipse;

	void OnCreate();
	void OnDestroy();
	void OnPaint();
	void OnResize();
	void OnLButtonDown(int pixelX, int pixelY, DWORD flags);
	void OnLButtonUp();
	void OnMouseMove(int pixelX, int pixelY, DWORD flags);

	void CalculateLayout();
	HRESULT CreateGraphicsResources();
	void DiscardGraphicsResources();
	void CaptureCursor();

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
};

