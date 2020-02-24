#include "Core.h"

LRESULT CALLBACK CoreProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	Core* pCore =
		reinterpret_cast<Core*>
		(GetWindowLongPtr(hwnd, GWLP_USERDATA));

	if (pCore) {
		return pCore->HandleMessage(hwnd, uMsg, wParam, lParam);
	}
	else {
		DestroyWindow(hwnd);
		PostQuitMessage(0);
	}
	return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

Core* Core::Create(HWND hWnd)
{
	return new Core(hWnd);
}

Core::Core(HWND hWnd) : 
	m_hWnd(hWnd), m_pStage(nullptr), 
	m_pStageEntities(nullptr), m_stageEntityCount(0), m_isActive(true)
{
	//RECT rc;
	//GetWindowRect(hWnd, &rc);
	//int screenResX = (GetSystemMetrics(SM_CXSCREEN) - rc.right)/2;
	//int screenResY = (GetSystemMetrics(SM_CYSCREEN) - rc.bottom)/2;

	//SetWindowPos(hWnd, 0, screenResX, screenResY, 0, 0, SWP_NOZORDER | SWP_NOSIZE);

	SetWindowLongPtr(
		hWnd, GWLP_USERDATA,
		reinterpret_cast<LONG_PTR>(this));
	SetWindowLongPtr(
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
		POINT ptMin = { rcClip.left, rcClip.top };
		POINT ptMax = { rcClip.right, rcClip.bottom };
		ClientToScreen(hWnd, &ptMin);
		ClientToScreen(hWnd, &ptMax);
		SetRect(&rcClip, ptMin.x, ptMin.y, ptMax.x, ptMax.y);
		ClipCursor(&rcClip);
		ShowCursor(false);
		int xOffset = (ptMax.x - ptMin.x);
		int yOffset = (ptMax.y - ptMin.y);
		int xCoord = ptMax.x - xOffset / 2;
		int yCoord = ptMax.y - yOffset / 2;
		SetCursorPos(xCoord, yCoord);
		m_pStage->GetMainCamera()->GetCamera()->SetMouseCoord(xCoord, yCoord);
		m_pStage->GetMainCamera()->GetCamera()->SetRotateStatus(true);
	}
	break;
	case WM_MOUSEMOVE:
	{
		if (m_pStage->GetMainCamera()->GetCamera()->GetRotateStatus()) {
			int xCoord = GET_X_LPARAM(lParam);
			int yCoord = GET_Y_LPARAM(lParam);
			POINT pt = { xCoord, yCoord };
			ClientToScreen(hWnd, &pt);
			int lastMouseX, lastMouseY;
			m_pStage->GetMainCamera()->GetCamera()->GetMouseCoord(lastMouseX, lastMouseY);
			m_pStage->GetMainCamera()->GetCamera()->SetMouseCoord(pt.x, pt.y);
			float offsetX = pt.x - lastMouseX;
			float offsetY = pt.y - lastMouseY;
			float pitch = offsetX * m_pStage->GetMainCamera()->GetCamera()->GetRotationSensitivity();
			float head = offsetY * m_pStage->GetMainCamera()->GetCamera()->GetRotationSensitivity();
			m_pStage->GetMainCamera()->GetTransform()->RotateEulerAngles(head, pitch, 0.0f);
		}
	}
	break;
	case WM_KEYDOWN:
	{
		using namespace DirectX;
		switch (wParam) {
		case 0x57: //W 
		{
			m_pStage->GetMainCamera()->GetTransform()->Translate(
				m_pStage->GetMainCamera()->GetCamera()->GetTranslationSpeed() * m_pStage->GetMainCamera()->GetTransform()->GetForwardDir());
		}
		break;
		case 0x41: //A
		{
			m_pStage->GetMainCamera()->GetTransform()->Translate(
				m_pStage->GetMainCamera()->GetCamera()->GetTranslationSpeed() * -1.0f * m_pStage->GetMainCamera()->GetTransform()->GetRightDir());
		}
		break;
		case 0x53: //S
		{
			m_pStage->GetMainCamera()->GetTransform()->Translate(
				m_pStage->GetMainCamera()->GetCamera()->GetTranslationSpeed() * -1.0f * m_pStage->GetMainCamera()->GetTransform()->GetForwardDir());
		}
		break;
		case 0x44: //D
		{
			m_pStage->GetMainCamera()->GetTransform()->Translate(
				m_pStage->GetMainCamera()->GetCamera()->GetTranslationSpeed() * m_pStage->GetMainCamera()->GetTransform()->GetRightDir());
		}
		break;
		case VK_ESCAPE:
		{
			ReleaseCapture();
			ClipCursor(nullptr);
			ShowCursor(true);
			m_pStage->GetMainCamera()->GetCamera()->SetRotateStatus(false);
			return DefWindowProc(hWnd, uMsg, wParam, lParam);
		}
		break;
		}
	}
	break;
	case WM_KILLFOCUS: 
	{
		ReleaseCapture();
		ShowCursor(true);
		m_pStage->GetMainCamera()->GetCamera()->SetRotateStatus(false);
		return DefWindowProc(hWnd, uMsg, wParam, lParam);
	}
	break;
	case WM_QUIT:
		DestroyWindow(hWnd);
	case WM_DESTROY:
	{
		m_isActive = false;
		PostQuitMessage(0);
	}
	break;
	default:
	{
		return DefWindowProc(hWnd, uMsg, wParam, lParam);
	}
	break;
	}
	return 0;
}

void Core::LoadStage(Stage& stage)
{
	m_pStage = new Stage(stage);
	m_stageEntityCount = m_pStage->GetEntityCount();
	m_pStageEntities = new Entity[m_stageEntityCount];
	memcpy(m_pStageEntities, stage.GetEntityCollection(), sizeof(Entity) * m_stageEntityCount);
}

bool Core::OnUpdate(Renderer& renderer)
{
	renderer.OnFrameRender(*m_pStage);
	return m_isActive;
}

void Core::Shutdown(void)
{
	SAFE_SHUTDOWN(m_pStage);
}

Stage* Core::GetStage()
{
	return m_pStage;
}
