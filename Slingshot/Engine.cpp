#include "Engine.h"

bool Engine::Initialize(WINDOW_DESC& window_desc, RENDERER_DESC& renderer_desc)
{
	if ((m_pWindow = Window::Create(window_desc)) != nullptr) {
		HWND hWnd = m_pWindow->GetHandle();
		m_isRunning = (m_pCore = Core::Create(hWnd)) != nullptr;
		m_isRunning = (m_pRenderer = Renderer::Create(hWnd, renderer_desc)) != nullptr;
	}

	if (m_isRunning) 
	{
		m_isRunning = m_pRenderer->Initialize();
	}
	
	if (m_isRunning) 
	{
		m_isRunning = EditStage(m_pStage);
		if (m_isRunning)
		{
			m_pCore->LoadStage(m_pStage);
		}
	}

	return m_isRunning;
}

bool Engine::EditStage(Stage* stage)
{
	bool success;

	Entity* entityCollection = new Entity[2];

	//Setup main camera
	//------------------------------
	RECT winRect;
	GetWindowRect(m_pWindow->GetHandle(), &winRect);
	float winWidth = static_cast<float>(winRect.right - winRect.left);
	float winHeight = static_cast<float>(winRect.bottom - winRect.top);

	entityCollection[0].SetCamera(
		&CAMERA_DESC(
			75.0f,
			winWidth,
			winHeight,
			1.0f, 1000.0f));
	//------------------------------


	//Setup rendering test triangle
	//------------------------------
	ColorShaderVertex* vertexCollection = new ColorShaderVertex[3];
	vertexCollection[0].position = DirectX::XMFLOAT4(0.0f, 0.5f, 1.0f, 1.0f);
	vertexCollection[0].color = DirectX::XMFLOAT4(1.0f, 0.0f, 0.0f, 1.0f);
	vertexCollection[1].position = DirectX::XMFLOAT4(0.5f, -0.5f, 1.0f, 1.0f);
	vertexCollection[1].color = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f);
	vertexCollection[2].position = DirectX::XMFLOAT4(-0.5f, -0.5f, 1.0f, 1.0f);
	vertexCollection[2].color = DirectX::XMFLOAT4(0.0f, 0.0f, 1.0f, 1.0f);

	unsigned int* indexCollection = new unsigned int[3];
	indexCollection[0] = 0;
	indexCollection[1] = 1;
	indexCollection[2] = 2;

	char* ColorVS_bytecode = nullptr, * ColorPS_bytecode = nullptr;
	size_t ColorVS_size, ColorPS_size;
	ColorVS_bytecode = GetFileBytecode("ColorVertexShader.cso", ColorVS_size);
	ColorPS_bytecode = GetFileBytecode("ColorPixelShader.cso", ColorPS_size);

	success = entityCollection[1].SetModel(
		m_pRenderer->GetGraphicsContext(),
		&MODEL_DESC(
			&MESH_DESC(
				VertexType::ColorShaderVertex,
				D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
				vertexCollection, 3,
				indexCollection, 3),
			&SHADER_DESC(
				ColorVS_bytecode, ColorVS_size,
				ColorPS_bytecode, ColorPS_size)));
	//------------------------------

	success = ((m_pStage = Stage::Create(0, &STAGE_DESC(entityCollection, 2, 0))) != nullptr);
	return success;
}

int Engine::Run()
{
	MSG msg = {};
	while (m_isRunning) 
	{
		while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) 
		{
			TranslateMessage(&msg);
			DispatchMessageW(&msg);
		}
		m_isRunning = m_pCore->OnUpdate(*m_pRenderer);
	}
	return (int)msg.wParam;
}

void Engine::Shutdown()
{
	SAFE_SHUTDOWN(m_pStage);
	SAFE_SHUTDOWN(m_pCore);
	SAFE_SHUTDOWN(m_pRenderer);
	SAFE_SHUTDOWN(m_pWindow);
}

Renderer* Engine::GetRenderer()
{
	return m_pRenderer;
}
