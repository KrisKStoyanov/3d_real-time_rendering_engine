#pragma once
#include "GraphicsContext.h"
#include "CUDAContextScheduler.cuh"

class CUDAGraphicsContext : public GraphicsContext
{
public:
	CUDAGraphicsContext();
	~CUDAGraphicsContext();

	virtual LRESULT HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam);

	virtual void OnCreate(HWND hwnd);
	virtual void OnDestroy();
	virtual void OnPaint();
	virtual void OnResize();
	virtual void OnLButtonDown(int pixelX, int pixelY, DWORD flags);
	virtual void OnLButtonUp();
	virtual void OnMouseMove(int pixelX, int pixelY, DWORD flags);

	HWND m_hWnd;
private:

};

