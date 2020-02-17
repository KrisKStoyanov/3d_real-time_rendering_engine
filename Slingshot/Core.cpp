#include "Core.h"

LRESULT CALLBACK CoreProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	Core* pCore =
		reinterpret_cast<Core*>
		(GetWindowLongPtrW(hwnd, GWLP_USERDATA));

	if (pCore) {
		return pCore->HandleMessage(hwnd, uMsg, wParam, lParam);
	}
	else {
		DestroyWindow(hwnd);
		PostQuitMessage(0);
	}
	return DefWindowProcW(hwnd, uMsg, wParam, lParam);
}

Core* Core::Create(HWND hWnd)
{
	return new Core(hWnd);
}

Core::Core(HWND hWnd) : 
	m_hWnd(hWnd), m_pStage(nullptr), 
	m_pStageEntities(nullptr), m_stageEntityCount(0),
	m_pRenderer(nullptr), m_isActive(true)
{
	SetWindowLongPtrW(
		hWnd, GWLP_USERDATA,
		reinterpret_cast<LONG_PTR>(this));
	SetWindowLongPtrW(
		hWnd, GWLP_WNDPROC,
		reinterpret_cast<LONG_PTR>(CoreProc));
}

LRESULT Core::HandleMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg) {
	case WM_LBUTTONDOWN:
	{
		SetCapture(hWnd);
		RECT rcClip;		
		GetClientRect(hWnd, &rcClip);
		POINT pt = { rcClip.left, rcClip.top };
		POINT pt2 = { rcClip.right, rcClip.bottom };
		ClientToScreen(hWnd, &pt);
		ClientToScreen(hWnd, &pt2);
		SetRect(&rcClip, pt.x, pt.y, pt2.x, pt2.y);
		ClipCursor(&rcClip);
		ShowCursor(false);
	}
	break;
	case WM_MOUSEMOVE:
	{
		int xPos = GET_X_LPARAM(lParam);
		int yPos = GET_Y_LPARAM(lParam);
		m_pStage->GetMainCamera()->GetTransform()->Rotate(static_cast<float>(xPos), static_cast<float>(yPos), 0.0f);
	}
	break;
	case WM_KEYDOWN:
	{
		switch (wParam) {
		case 0x57: //W 
		{
			m_pStage->GetMainCamera()->GetTransform()->Translate(DirectX::XMVectorSet(0.0f, 0.0, 1.0, 0.0f));
		}
		break;
		case 0x41: //A
		{
			m_pStage->GetMainCamera()->GetTransform()->Translate(DirectX::XMVectorSet(-1.0f, 0.0, 0.0, 0.0f));
		}
		break;
		case 0x53: //S
		{
			m_pStage->GetMainCamera()->GetTransform()->Translate(DirectX::XMVectorSet(0.0f, 0.0, -1.0, 0.0f));
		}
		break;
		case 0x44: //D
		{
			m_pStage->GetMainCamera()->GetTransform()->Translate(DirectX::XMVectorSet(1.0f, 0.0, 0.0, 0.0f));
		}
		break;
		case VK_ESCAPE:
		{
			DestroyWindow(hWnd);
		}
		break;
		}
	}
	break;
	case WM_KILLFOCUS: 
	{
		ReleaseCapture();
		ShowCursor(true);
		return DefWindowProcW(hWnd, uMsg, wParam, lParam);
	}
	break;
	case WM_QUIT:
	case WM_DESTROY:
	{
		m_isActive = false;
		PostQuitMessage(0);
	}
	break;
	default:
	{
		return DefWindowProcW(hWnd, uMsg, wParam, lParam);
	}
	break;
	}
	return 0;
}

bool Core::InitializeRenderer(HWND hWnd, RENDERER_DESC* renderer_desc)
{
	m_pRenderer = Renderer::Create(hWnd, renderer_desc);
	bool success = (m_pRenderer != nullptr);
	if (success) {
		success = m_pRenderer->Initialize();
	}
	return success;
}

void Core::LoadStage(Stage* stage)
{
	m_pStage = stage;
	m_pStageEntities = m_pStage->GetEntityCollection();
	m_stageEntityCount = m_pStage->GetEntityCount();
}

bool Core::OnUpdate(void)
{
	m_pRenderer->OnFrameRender(m_pStage);
	return m_isActive;
}

void Core::Shutdown(void)
{
	SAFE_SHUTDOWN(m_pRenderer);
}

Renderer* Core::GetRenderer()
{
	return m_pRenderer;
}

Stage* Core::GetStage()
{
	return m_pStage;
}
