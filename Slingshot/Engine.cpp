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
			m_pCore->LoadStage(*m_pStage);
		}
	}

	return m_isRunning;
}

bool Engine::EditStage(Stage* stage)
{
	bool success;

	Entity* entityCollection = new Entity[2];

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
	//------------------------------


	//Triangle Object
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

	MESH_DESC mesh_desc;
	mesh_desc.vertexType = VertexType::ColorShaderVertex;
	mesh_desc.topology = D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
	mesh_desc.vertexCollection = new ColorShaderVertex[3];
	memcpy(mesh_desc.vertexCollection, vertexCollection, sizeof(ColorShaderVertex)* 3);
	mesh_desc.vertexCount = 3;
	mesh_desc.indexCollection = new unsigned int[3];
	memcpy(mesh_desc.indexCollection, indexCollection, sizeof(unsigned int) * 3);
	mesh_desc.indexCount = 3;

	SHADER_DESC shader_desc;
	shader_desc.VS_bytecode = new char[ColorVS_size];
	memcpy(shader_desc.VS_bytecode, ColorVS_bytecode, ColorVS_size);
	shader_desc.VS_size = ColorVS_size;
	shader_desc.PS_bytecode = new char[ColorPS_size];
	memcpy(shader_desc.PS_bytecode, ColorPS_bytecode, ColorPS_size);
	shader_desc.PS_size = ColorPS_size;

	MODEL_DESC model_desc;
	model_desc.mesh_desc = mesh_desc;
	model_desc.shader_desc = shader_desc;

	success = entityCollection[1].SetModel(
		*m_pRenderer->GetGraphicsContext(), model_desc);
	//------------------------------
	STAGE_DESC stage_desc;
	stage_desc.entityCount = 2;
	stage_desc.mainCameraId = 0;
	success = ((m_pStage = Stage::Create(0, stage_desc, *entityCollection)) != nullptr);

	SAFE_DELETE_ARRAY(vertexCollection);
	SAFE_DELETE_ARRAY(indexCollection);
	SAFE_DELETE_ARRAY(ColorVS_bytecode);
	SAFE_DELETE_ARRAY(ColorPS_bytecode);

	SAFE_DELETE_ARRAY(entityCollection);

	return success;
}

int Engine::Run()
{
	MSG msg = {};
	while (m_isRunning) 
	{
		while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) 
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
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
