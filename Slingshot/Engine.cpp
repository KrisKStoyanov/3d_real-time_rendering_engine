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

	const int ENTITY_COUNT = 3;
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

	TRANSFORM_DESC transform_desc;
	transform_desc.position = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f);

	entityCollection[0].SetCamera(camera_desc);
	entityCollection[0].SetTransform(transform_desc);
	//------------------------------

	//Color Shader
	//------------------------------
	char* ColorVS_bytecode = nullptr, * ColorPS_bytecode = nullptr;
	size_t ColorVS_size, ColorPS_size;
	ColorVS_bytecode = GetFileBytecode("ColorVertexShader.cso", ColorVS_size);
	ColorPS_bytecode = GetFileBytecode("ColorPixelShader.cso", ColorPS_size);
	//------------------------------

	//Triangle Object
	//------------------------------
	const int ENTITY0_VERTEX_COUNT = 3;
	const int ENTITY0_INDEX_COUNT = 3;

	ColorShaderVertex* triV_Collection = new ColorShaderVertex[ENTITY0_VERTEX_COUNT];
	triV_Collection[0].position = DirectX::XMFLOAT4(0.0f, 1.5f, 5.0f, 1.0f);
	triV_Collection[0].color = DirectX::XMFLOAT4(1.0f, 0.0f, 0.0f, 1.0f);
	triV_Collection[1].position = DirectX::XMFLOAT4(0.5f, 0.5f, 5.0f, 1.0f);
	triV_Collection[1].color = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f);
	triV_Collection[2].position = DirectX::XMFLOAT4(-0.5f, 0.5f, 5.0f, 1.0f);
	triV_Collection[2].color = DirectX::XMFLOAT4(0.0f, 0.0f, 1.0f, 1.0f);

	unsigned int* triI_Collection = new unsigned int[ENTITY0_INDEX_COUNT];
	triI_Collection[0] = 0;
	triI_Collection[1] = 1;
	triI_Collection[2] = 2;

	MESH_DESC triM_desc;
	triM_desc.vertexType = VertexType::ColorShaderVertex;
	triM_desc.topology = D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
	triM_desc.vertexCollection = new ColorShaderVertex[ENTITY0_VERTEX_COUNT];
	memcpy(triM_desc.vertexCollection, triV_Collection, sizeof(ColorShaderVertex) * ENTITY0_VERTEX_COUNT);
	triM_desc.vertexCount = ENTITY0_VERTEX_COUNT;
	triM_desc.indexCollection = new unsigned int[ENTITY0_INDEX_COUNT];
	memcpy(triM_desc.indexCollection, triI_Collection, sizeof(unsigned int) * ENTITY0_INDEX_COUNT);
	triM_desc.indexCount = ENTITY0_INDEX_COUNT;

	SHADER_DESC triS_desc;
	triS_desc.VS_bytecode = new char[ColorVS_size];
	memcpy(triS_desc.VS_bytecode, ColorVS_bytecode, ColorVS_size);
	triS_desc.VS_size = ColorVS_size;
	triS_desc.PS_bytecode = new char[ColorPS_size];
	memcpy(triS_desc.PS_bytecode, ColorPS_bytecode, ColorPS_size);
	triS_desc.PS_size = ColorPS_size;

	MODEL_DESC triModel_desc;
	triModel_desc.mesh_desc = triM_desc;
	triModel_desc.shader_desc = triS_desc;

	success = entityCollection[1].SetModel(
		*m_pRenderer->GetGraphicsContext(), triModel_desc);
	//------------------------------

	//Ground Object
	//------------------------------
	const int ENTITY1_VERTEX_COUNT = 4;
	const int ENTITY1_INDEX_COUNT = 5;

	ColorShaderVertex* groundV_Collection = new ColorShaderVertex[ENTITY1_VERTEX_COUNT];
	groundV_Collection[0].position = DirectX::XMFLOAT4(-20.0f, -3.0f, 40.0f, 1.0f);
	groundV_Collection[0].color = DirectX::XMFLOAT4(0.0f, 0.4f, 0.3f, 1.0f);
	groundV_Collection[1].position = DirectX::XMFLOAT4(20.0f, -3.0f, 0.0f, 1.0f);
	groundV_Collection[1].color = DirectX::XMFLOAT4(0.0f, 0.4f, 0.3f, 1.0f);
	groundV_Collection[2].position = DirectX::XMFLOAT4(-20.0f, -3.0f, 0.0f, 1.0f);
	groundV_Collection[2].color = DirectX::XMFLOAT4(0.0f, 0.4f, 0.3f, 1.0f);
	groundV_Collection[3].position = DirectX::XMFLOAT4(20.0f, -3.0f, 40.0f, 1.0f);
	groundV_Collection[3].color = DirectX::XMFLOAT4(0.0f, 0.4f, 0.3f, 1.0f);

	unsigned int* groundI_Collection = new unsigned int[ENTITY1_INDEX_COUNT];
	groundI_Collection[0] = 0;
	groundI_Collection[1] = 3;
	groundI_Collection[2] = 1;
	groundI_Collection[3] = 2;
	groundI_Collection[4] = 0;

	MESH_DESC groundM_desc;
	groundM_desc.vertexType = VertexType::ColorShaderVertex;
	groundM_desc.topology = D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
	groundM_desc.vertexCollection = new ColorShaderVertex[ENTITY1_VERTEX_COUNT];
	memcpy(groundM_desc.vertexCollection, groundV_Collection, sizeof(ColorShaderVertex) * ENTITY1_VERTEX_COUNT);
	groundM_desc.vertexCount = ENTITY1_VERTEX_COUNT;
	groundM_desc.indexCollection = new unsigned int[ENTITY1_INDEX_COUNT];
	memcpy(groundM_desc.indexCollection, groundI_Collection, sizeof(unsigned int) * ENTITY1_INDEX_COUNT);
	groundM_desc.indexCount = ENTITY1_INDEX_COUNT;

	SHADER_DESC groundS_desc;
	groundS_desc.VS_bytecode = new char[ColorVS_size];
	memcpy(groundS_desc.VS_bytecode, ColorVS_bytecode, ColorVS_size);
	groundS_desc.VS_size = ColorVS_size;
	groundS_desc.PS_bytecode = new char[ColorPS_size];
	memcpy(groundS_desc.PS_bytecode, ColorPS_bytecode, ColorPS_size);
	groundS_desc.PS_size = ColorPS_size;

	MODEL_DESC groundModel_desc;
	groundModel_desc.mesh_desc = groundM_desc;
	groundModel_desc.shader_desc = groundS_desc;

	success = entityCollection[2].SetModel(
		*m_pRenderer->GetGraphicsContext(), groundModel_desc);
	//------------------------------

	STAGE_DESC stage_desc;
	stage_desc.entityCollection = new Entity[ENTITY_COUNT];
	memcpy(stage_desc.entityCollection, entityCollection, sizeof(Entity) * ENTITY_COUNT);
	stage_desc.entityCount = ENTITY_COUNT;
	stage_desc.mainCameraId = 0;
	success = ((m_pStage = Stage::Create(0, stage_desc)) != nullptr);

	SAFE_DELETE_ARRAY(triV_Collection);
	SAFE_DELETE_ARRAY(triI_Collection);
	SAFE_DELETE_ARRAY(groundV_Collection);
	SAFE_DELETE_ARRAY(groundI_Collection);
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
