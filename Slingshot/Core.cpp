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
		POINT pt = { rcClip.left, rcClip.top };
		POINT pt2 = { rcClip.right, rcClip.bottom };
		ClientToScreen(hWnd, &pt);
		ClientToScreen(hWnd, &pt2);
		SetRect(&rcClip, pt.x, pt.y, pt2.x, pt2.y);
		ClipCursor(&rcClip);
		ShowCursor(false);
		float xCoord = static_cast<float>(rcClip.right - rcClip.left) / 2.0f;
		float yCoord = static_cast<float>(rcClip.bottom - rcClip.top) / 2.0f;
		SetCursorPos(xCoord, yCoord);
		m_pStage->GetMainCamera()->GetCamera()->SetMouseCoord(0.0f, 0.0f);
		m_pStage->GetMainCamera()->GetCamera()->SetRotateStatus(true);
	}
	break;
	case WM_MOUSEMOVE:
	{
		if (m_pStage->GetMainCamera()->GetCamera()->GetRotateStatus()) {
			float xCoord = static_cast<float>(GET_X_LPARAM(lParam));
			float yCoord = static_cast<float>(GET_Y_LPARAM(lParam));
			float lastMouseX, lastMouseY;
			m_pStage->GetMainCamera()->GetCamera()->GetMouseCoord(lastMouseX, lastMouseY);
			m_pStage->GetMainCamera()->GetCamera()->SetMouseCoord(xCoord, yCoord);
			float offsetX = xCoord - lastMouseX;
			float offsetY = yCoord - lastMouseY;
			float pitch = offsetX * m_pStage->GetMainCamera()->GetCamera()->GetRotationSensitivity();
			float head = offsetY * m_pStage->GetMainCamera()->GetCamera()->GetRotationSensitivity();
			m_pStage->GetMainCamera()->GetTransform()->Rotate(-head, pitch, 0.0f);
		}
	}
	break;
	case WM_KEYDOWN:
	{
		switch (wParam) {
		case 0x57: //W 
		{
			m_pStage->GetMainCamera()->GetTransform()->Translate(m_pStage->GetMainCamera()->GetTransform()->GetForwardDir());
		}
		break;
		case 0x41: //A
		{
			m_pStage->GetMainCamera()->GetTransform()->Translate(m_pStage->GetMainCamera()->GetTransform()->GetRightDir());
		}
		break;
		case 0x53: //S
		{
			using namespace DirectX;
			m_pStage->GetMainCamera()->GetTransform()->Translate(-1.0f * m_pStage->GetMainCamera()->GetTransform()->GetForwardDir());
		}
		break;
		case 0x44: //D - Currently incorrect X-Axis movement due to misalignment post-mouse rotation (Correct: -1.0f * [A] translation)
		{
			using namespace DirectX;
			m_pStage->GetMainCamera()->GetTransform()->Translate(-1.0f * m_pStage->GetMainCamera()->GetTransform()->GetRightDir());
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
		m_pStage->GetMainCamera()->GetCamera()->SetRotateStatus(false);
		return DefWindowProc(hWnd, uMsg, wParam, lParam);
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
