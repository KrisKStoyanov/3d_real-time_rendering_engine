#include "Engine.h"

LRESULT CALLBACK EngineProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	Engine* pEngine =
		reinterpret_cast<Engine*>
		(GetWindowLongPtr(hwnd, GWLP_USERDATA));

	if (pEngine) {
		return pEngine->HandleWindowMessage(hwnd, uMsg, wParam, lParam);
	}
	else {
		DestroyWindow(hwnd);
		PostQuitMessage(0);
	}
	return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

bool Engine::Initialize(WINDOW_DESC& window_desc, RENDERER_DESC& renderer_desc)
{
	if ((m_pWindow = Window::Create(window_desc)) != nullptr) {

		HWND hWnd = m_pWindow->GetHandle();

		SetWindowLongPtr(
			hWnd, GWLP_USERDATA,
			reinterpret_cast<LONG_PTR>(this));
		SetWindowLongPtr(
			hWnd, GWLP_WNDPROC,
			reinterpret_cast<LONG_PTR>(EngineProc));

		m_isRunning = (m_pRenderer = Renderer::Create(hWnd, renderer_desc)) != nullptr;
	}

	if (m_isRunning) 
	{
		m_isRunning = m_pRenderer->Initialize();
		EditStage(*m_pStage);
		m_pTimer = new Timer();
	}

	return m_isRunning;
}

void Engine::EditStage(Stage& stage)
{
	//Pipeline State
	//------------------------------
	PIPELINE_DESC pipeline_desc;
	pipeline_desc.VS_filename = "ColorVertexShader.cso";
	pipeline_desc.PS_filename = "ColorPixelShader.cso";

	m_pRenderer->SetPipelineState(pipeline_desc, VertexType::ColorShaderVertex);
	//------------------------------

	const int ENTITY_COUNT = 8;
	Entity* entityCollection = new Entity[ENTITY_COUNT];

	//Main Camera
	//------------------------------
	RECT winRect;
	GetWindowRect(m_pWindow->GetHandle(), &winRect);
	float winWidth = static_cast<float>(winRect.right - winRect.left);
	float winHeight = static_cast<float>(winRect.bottom - winRect.top);

	CAMERA_DESC camera_desc;
	camera_desc.lenseWidth = winWidth;
	camera_desc.lenseHeight = winHeight;
	entityCollection[0].SetCamera(camera_desc);

	TRANSFORM_DESC entity0_transform_desc;
	entity0_transform_desc.position = DirectX::XMFLOAT4(0.0f, 5.0f, -15.0f, 1.0f);
	entityCollection[0].SetTransform(entity0_transform_desc);
	
	// CORNEL BOX
	//------------------------------
	//Bottom
	TRANSFORM_DESC entity1_transform_desc;
	entity1_transform_desc.position = DirectX::XMFLOAT4(0.0f, -5.0f, 10.0f, 1.0f);
	entityCollection[1].SetTransform(entity1_transform_desc);
	CreatePlane(*(entityCollection + 1), 10.0f, 10.0f, DirectX::XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f));

	//Front
	TRANSFORM_DESC entity2_transform_desc;
	entity2_transform_desc.position = DirectX::XMFLOAT4(0.0f, 5.0f, 20.0f, 1.0f);
	entity2_transform_desc.rotation = DirectX::XMFLOAT4(-90.0f, 0.0f, 0.0f, 0.0f);
	entityCollection[2].SetTransform(entity2_transform_desc);
	CreatePlane(*(entityCollection + 2), 10.0f, 10.0f, DirectX::XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f));

	//Left
	TRANSFORM_DESC entity3_transform_desc;
	entity3_transform_desc.position = DirectX::XMFLOAT4(-10.0f, 5.0f, 10.0f, 1.0f);
	entity3_transform_desc.rotation = DirectX::XMFLOAT4(0.0f, 0.0f, -90.0f, 0.0f);
	entityCollection[3].SetTransform(entity3_transform_desc);
	CreatePlane(*(entityCollection + 3), 10.0f, 10.0f, DirectX::XMFLOAT4(1.0f, 0.0f, 0.0f, 1.0f));

	//Right
	TRANSFORM_DESC entity4_transform_desc;
	entity4_transform_desc.position = DirectX::XMFLOAT4(10.0f, 5.0f, 10.0f, 1.0f);
	entity4_transform_desc.rotation = DirectX::XMFLOAT4(0.0f, 0.0f, 90.0f, 0.0f);
	entityCollection[4].SetTransform(entity4_transform_desc);
	CreatePlane(*(entityCollection + 4), 10.0f, 10.0f, DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f));

	//Top
	TRANSFORM_DESC entity5_transform_desc;
	entity5_transform_desc.position = DirectX::XMFLOAT4(0.0f, 15.0f, 10.0f, 1.0f);
	entity5_transform_desc.rotation = DirectX::XMFLOAT4(-180.0f, 0.0f, 0.0f, 0.0f);
	entityCollection[5].SetTransform(entity5_transform_desc);
	CreatePlane(*(entityCollection + 5), 10.0f, 10.0f, DirectX::XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f));
	//------------------------------

	// DECOR
	//------------------------------
	TRANSFORM_DESC entity6_transform_desc;
	entity6_transform_desc.position = DirectX::XMFLOAT4(0.0f, 2.0f, 10.0f, 1.0f);
	entityCollection[6].SetTransform(entity6_transform_desc);
	CreateCube(*(entityCollection + 6), 2.0f, 2.0f, 2.0f, DirectX::XMFLOAT4(0.9f, 0.9f, 0.9f, 1.0f));

	TRANSFORM_DESC entity7_transform_desc;
	entity7_transform_desc.position = DirectX::XMFLOAT4(-7.5f, 2.0f, 10.0f, 1.0f);
	entityCollection[7].SetTransform(entity7_transform_desc);
	CreateSphere(*(entityCollection + 7), 30, 30, 2, DirectX::XMFLOAT4(0.7f, 0.7f, 0.7f, 1.0f));
	//------------------------------

	STAGE_DESC stage_desc;
	stage_desc.entityCollection = new Entity[ENTITY_COUNT];
	memcpy(stage_desc.entityCollection, entityCollection, sizeof(Entity) * ENTITY_COUNT);
	stage_desc.entityCount = ENTITY_COUNT;
	stage_desc.mainCameraId = 0;
	m_pStage = Stage::Create(0, stage_desc);

	SAFE_DELETE_ARRAY(entityCollection);
}

int Engine::Run()
{
	MSG msg = {};
	while (m_isRunning) 
	{
		m_pTimer->OnFrameStart();
		while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) 
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}

		m_pRenderer->OnFrameRender(*m_pStage);
	}
	return (int)msg.wParam;
}

void Engine::Shutdown()
{
	SAFE_SHUTDOWN(m_pStage);
	SAFE_SHUTDOWN(m_pRenderer);
	SAFE_SHUTDOWN(m_pWindow);
	SAFE_DELETE(m_pTimer);
}

void Engine::CreatePlane(Entity& entity, float width, float length, DirectX::XMFLOAT4 color)
{
	//Plane Object
	//------------------------------
	const int VERTEX_COUNT = 4;
	const int INDEX_COUNT = 4;

	ColorShaderVertex* groundV_Collection = new ColorShaderVertex[VERTEX_COUNT];
	groundV_Collection[0].position = DirectX::XMFLOAT4(-1.0f * width, 0.0f, -1.0f * length, 1.0f);
	groundV_Collection[0].color = color;
	groundV_Collection[1].position = DirectX::XMFLOAT4(-1.0f * width, 0.0f, 1.0f * length, 1.0f);
	groundV_Collection[1].color = color;
	groundV_Collection[2].position = DirectX::XMFLOAT4(1.0f * width, 0.0f, -1.0f * length, 1.0f);
	groundV_Collection[2].color = color;
	groundV_Collection[3].position = DirectX::XMFLOAT4(1.0f * width, 0.0f, 1.0f * length, 1.0f);
	groundV_Collection[3].color = color;

	unsigned int* groundI_Collection = new unsigned int[INDEX_COUNT];
	groundI_Collection[0] = 0;
	groundI_Collection[1] = 1;
	groundI_Collection[2] = 2;
	groundI_Collection[3] = 3;

	MESH_DESC groundM_desc;
	groundM_desc.topology = D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
	groundM_desc.vertexCollection = new ColorShaderVertex[VERTEX_COUNT];
	memcpy(groundM_desc.vertexCollection, groundV_Collection, sizeof(ColorShaderVertex) * VERTEX_COUNT);
	groundM_desc.vertexCount = VERTEX_COUNT;
	groundM_desc.indexCollection = new unsigned int[INDEX_COUNT];
	memcpy(groundM_desc.indexCollection, groundI_Collection, sizeof(unsigned int) * INDEX_COUNT);
	groundM_desc.indexCount = INDEX_COUNT;

	entity.SetModel(
		*m_pRenderer->GetGraphicsContext(), groundM_desc, VertexType::ColorShaderVertex);

	SAFE_DELETE_ARRAY(groundV_Collection);
	SAFE_DELETE_ARRAY(groundI_Collection);
}

void Engine::CreateCube(Entity& entity, float width, float height, float length, DirectX::XMFLOAT4 color)
{
	//Cube Object
	//------------------------------
	const int VERTEX_COUNT = 8;
	const int INDEX_COUNT = 14;

	ColorShaderVertex* cubeV_collection = new ColorShaderVertex[VERTEX_COUNT];
	cubeV_collection[0].position = DirectX::XMFLOAT4(-1.0f * width, -1.0f * height, -1.0f * length, 1.0f);
	cubeV_collection[0].color = color;
	cubeV_collection[1].position = DirectX::XMFLOAT4(-1.0f * width, 1.0f * height, -1.0f * length, 1.0f);
	cubeV_collection[1].color = color;
	cubeV_collection[2].position = DirectX::XMFLOAT4(1.0f * width, -1.0f * height, -1.0f * length, 1.0f);
	cubeV_collection[2].color = color;
	cubeV_collection[3].position = DirectX::XMFLOAT4(1.0f * width, 1.0f * height, -1.0f * length, 1.0f);
	cubeV_collection[3].color = color;
	cubeV_collection[4].position = DirectX::XMFLOAT4(-1.0f * width, -1.0f * height, 1.0f * length, 1.0f);
	cubeV_collection[4].color = color;
	cubeV_collection[5].position = DirectX::XMFLOAT4(-1.0f * width, 1.0f * height, 1.0f * length, 1.0f);
	cubeV_collection[5].color = color;
	cubeV_collection[6].position = DirectX::XMFLOAT4(1.0f * width, -1.0f * height, 1.0f * length, 1.0f);
	cubeV_collection[6].color = color;
	cubeV_collection[7].position = DirectX::XMFLOAT4(1.0f * width, 1.0f * height, 1.0f * length, 1.0f);
	cubeV_collection[7].color = color;

	unsigned int* cubeI_collection = new unsigned int[INDEX_COUNT];

	//Front
	cubeI_collection[0] = 0;
	cubeI_collection[1] = 1;
	cubeI_collection[2] = 2;
	cubeI_collection[3] = 3;

	cubeI_collection[4] = 7;
	cubeI_collection[5] = 1;

	cubeI_collection[6] = 5;
	cubeI_collection[7] = 0;

	cubeI_collection[8] = 4;
	cubeI_collection[9] = 2;

	cubeI_collection[10] = 6;
	cubeI_collection[11] = 7;

	cubeI_collection[12] = 4;
	cubeI_collection[13] = 5;

	MESH_DESC cubeM_desc;
	cubeM_desc.topology = D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
	cubeM_desc.vertexCollection = new ColorShaderVertex[VERTEX_COUNT];
	memcpy(cubeM_desc.vertexCollection, cubeV_collection, sizeof(ColorShaderVertex) * VERTEX_COUNT);
	cubeM_desc.vertexCount = VERTEX_COUNT;
	cubeM_desc.indexCollection = new unsigned int[INDEX_COUNT];
	memcpy(cubeM_desc.indexCollection, cubeI_collection, sizeof(unsigned int) * INDEX_COUNT);
	cubeM_desc.indexCount = INDEX_COUNT;

	entity.SetModel(
		*m_pRenderer->GetGraphicsContext(), cubeM_desc, VertexType::ColorShaderVertex);

	SAFE_DELETE_ARRAY(cubeV_collection);
	SAFE_DELETE_ARRAY(cubeI_collection);
}

void Engine::CreateSphere(Entity& entity, unsigned int slices, unsigned int stacks, float radius, DirectX::XMFLOAT4 color)
{
	const int VERTEX_COUNT = (stacks + 1) * (slices + 1);
	const int INDEX_COUNT = (slices * stacks + slices) * 6;

	ColorShaderVertex* sphereV_collection = new ColorShaderVertex[VERTEX_COUNT];
	unsigned int* sphereI_collection = new unsigned int[INDEX_COUNT];
	
	float slicesF = static_cast<float>(slices);
	float stacksF = static_cast<float>(stacks);

	for (unsigned int i = 0; i < stacks + 1; ++i) 
	{
		float V = i / stacksF;
		float phi = V * 3.14f;

		for (unsigned int j = 0; j < slices + 1; ++j) {

			float U = j / slicesF;
			float theta = U * 6.28f;

			float sinPhi = sinf(phi);

			float x = cosf(theta) * sinPhi;
			float y = cosf(phi);
			float z = sinf(theta) * sinPhi;

			int index = j + i * (slices + 1);
			sphereV_collection[index].position = DirectX::XMFLOAT4(x * radius, y * radius, z * radius, 1.0f);
			sphereV_collection[index].color = color;

			//vert.normal = glm::vec3(x, y, z);
			//vert.uv = glm::vec2((glm::asin(vert.normal.x) / piVal + 0.5f), (glm::asin(vert.normal.y) / piVal + 0.5f));
		}
	}

	int index = 0;
	for (unsigned int i = 0; i < slices * stacks + slices; ++i) 
	{
		sphereI_collection[index] = i;
		sphereI_collection[index + 1] = i + slices + 1;
		sphereI_collection[index + 2] = i + slices;
		sphereI_collection[index + 3] = i + slices + 1;
		sphereI_collection[index + 4] = i;
		sphereI_collection[index + 5] = i + 1;

		index += 6;
	}

	MESH_DESC sphereM_desc;
	sphereM_desc.topology = D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
	sphereM_desc.vertexCollection = new ColorShaderVertex[VERTEX_COUNT];
	memcpy(sphereM_desc.vertexCollection, sphereV_collection, sizeof(ColorShaderVertex) * VERTEX_COUNT);
	sphereM_desc.vertexCount = VERTEX_COUNT;
	sphereM_desc.indexCollection = new unsigned int[INDEX_COUNT];
	memcpy(sphereM_desc.indexCollection, sphereI_collection, sizeof(unsigned int) * INDEX_COUNT);
	sphereM_desc.indexCount = INDEX_COUNT;

	entity.SetModel(
		*m_pRenderer->GetGraphicsContext(), sphereM_desc, VertexType::ColorShaderVertex);

	SAFE_DELETE_ARRAY(sphereV_collection);
	SAFE_DELETE_ARRAY(sphereI_collection);
}

LRESULT Engine::HandleWindowMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
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
			float offsetX = static_cast<float>(pt.x) - lastMouseX;
			float offsetY = static_cast<float>(pt.y) - lastMouseY;
			float pitch = offsetX * m_pStage->GetMainCamera()->GetCamera()->GetRotationSpeed();
			float head = offsetY * m_pStage->GetMainCamera()->GetCamera()->GetRotationSpeed();
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
				m_pStage->GetMainCamera()->GetCamera()->GetTranslationSpeed() * 
				m_pStage->GetMainCamera()->GetTransform()->GetForwardDir() * 
				m_pTimer->m_smoothstepF);
		}
		break;
		case 0x41: //A
		{
			m_pStage->GetMainCamera()->GetTransform()->Translate(
				m_pStage->GetMainCamera()->GetCamera()->GetTranslationSpeed() * -1.0f * 
				m_pStage->GetMainCamera()->GetTransform()->GetRightDir() * 
				m_pTimer->m_smoothstepF);
		}
		break;
		case 0x53: //S
		{
			m_pStage->GetMainCamera()->GetTransform()->Translate(
				m_pStage->GetMainCamera()->GetCamera()->GetTranslationSpeed() * -1.0f * 
				m_pStage->GetMainCamera()->GetTransform()->GetForwardDir() * 
				m_pTimer->m_smoothstepF);
		}
		break;
		case 0x44: //D
		{
			m_pStage->GetMainCamera()->GetTransform()->Translate(
				m_pStage->GetMainCamera()->GetCamera()->GetTranslationSpeed() * 
				m_pStage->GetMainCamera()->GetTransform()->GetRightDir() * 
				m_pTimer->m_smoothstepF);
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
		m_isRunning = false;
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

Renderer* Engine::GetRenderer()
{
	return m_pRenderer;
}
