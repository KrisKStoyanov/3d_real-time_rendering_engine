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

	TRANSFORM_DESC mc_transform_desc;
	mc_transform_desc.position = DirectX::XMFLOAT4(0.0f, 1.0f, -5.0f, 1.0f);

	entityCollection[0].SetCamera(camera_desc);
	entityCollection[0].SetTransform(mc_transform_desc);
	//------------------------------

	//Color Shader
	//------------------------------
	char* ColorVS_bytecode = nullptr, * ColorPS_bytecode = nullptr;
	size_t ColorVS_size, ColorPS_size;
	ColorVS_bytecode = GetFileBytecode("ColorVertexShader.cso", ColorVS_size);
	ColorPS_bytecode = GetFileBytecode("ColorPixelShader.cso", ColorPS_size);
	//------------------------------

	//Cube Object
	//------------------------------
	const int ENTITY0_VERTEX_COUNT = 8;
	const int ENTITY0_INDEX_COUNT = 18;

	ColorShaderVertex* cubeV_collection = new ColorShaderVertex[ENTITY0_VERTEX_COUNT];
	cubeV_collection[0].position = DirectX::XMFLOAT4(-2.0f, -2.0f, -2.0f, 1.0f);
	cubeV_collection[0].color = DirectX::XMFLOAT4(1.0f, 0.0f, 0.0f, 1.0f);
	cubeV_collection[1].position = DirectX::XMFLOAT4(-2.0f, 2.0f, -2.0f, 1.0f);
	cubeV_collection[1].color = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f);
	cubeV_collection[2].position = DirectX::XMFLOAT4(2.0f, -2.0f, -2.0f, 1.0f);
	cubeV_collection[2].color = DirectX::XMFLOAT4(0.0f, 0.0f, 1.0f, 1.0f);
	cubeV_collection[3].position = DirectX::XMFLOAT4(2.0f, 2.0f, -2.0f, 1.0f);
	cubeV_collection[3].color = DirectX::XMFLOAT4(1.0f, 0.0f, 0.0f, 1.0f);
	cubeV_collection[4].position = DirectX::XMFLOAT4(-2.0f, -2.0f, 2.0f, 1.0f);
	cubeV_collection[4].color = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f);
	cubeV_collection[5].position = DirectX::XMFLOAT4(-2.0f, 2.0f, 2.0f, 1.0f);
	cubeV_collection[5].color = DirectX::XMFLOAT4(0.0f, 0.0f, 1.0f, 1.0f);
	cubeV_collection[6].position = DirectX::XMFLOAT4(2.0f, -2.0f, 2.0f, 1.0f);
	cubeV_collection[6].color = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f);
	cubeV_collection[7].position = DirectX::XMFLOAT4(2.0f, 2.0f, 2.0f, 1.0f);
	cubeV_collection[7].color = DirectX::XMFLOAT4(0.0f, 0.0f, 1.0f, 1.0f);

	unsigned int* cubeI_collection = new unsigned int[ENTITY0_INDEX_COUNT];

	//Front
	cubeI_collection[0] = 0;
	cubeI_collection[1] = 1;
	cubeI_collection[2] = 2;
	cubeI_collection[3] = 3;

	//Right
	cubeI_collection[4] = 6;
	cubeI_collection[5] = 7;

	//Back
	cubeI_collection[6] = 4;
	cubeI_collection[7] = 5;

	//Left
	cubeI_collection[8] = 0;
	cubeI_collection[9] = 1;

	//Top
	cubeI_collection[10] = 1; //Degenerate: switches winding order
	cubeI_collection[11] = 5;
	cubeI_collection[12] = 3;
	cubeI_collection[13] = 7;

	//Bottom
	cubeI_collection[14] = 2; //Degenerate: switches winding order
	cubeI_collection[15] = 6; //Degenerate: switches winding order
	cubeI_collection[16] = 0;
	cubeI_collection[17] = 4;

	MESH_DESC triM_desc;
	triM_desc.vertexType = VertexType::ColorShaderVertex;
	triM_desc.topology = D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
	triM_desc.vertexCollection = new ColorShaderVertex[ENTITY0_VERTEX_COUNT];
	memcpy(triM_desc.vertexCollection, cubeV_collection, sizeof(ColorShaderVertex) * ENTITY0_VERTEX_COUNT);
	triM_desc.vertexCount = ENTITY0_VERTEX_COUNT;
	triM_desc.indexCollection = new unsigned int[ENTITY0_INDEX_COUNT];
	memcpy(triM_desc.indexCollection, cubeI_collection, sizeof(unsigned int) * ENTITY0_INDEX_COUNT);
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

	TRANSFORM_DESC triT_desc;
	triT_desc.position = DirectX::XMFLOAT4(0.0f, 2.0f, 20.0f, 1.0f);
	entityCollection[1].SetTransform(triT_desc);

	success = entityCollection[1].SetModel(
		*m_pRenderer->GetGraphicsContext(), triModel_desc);
	//------------------------------

	//Ground Object
	//------------------------------
	const int ENTITY1_VERTEX_COUNT = 4;
	const int ENTITY1_INDEX_COUNT = 4;

	ColorShaderVertex* groundV_Collection = new ColorShaderVertex[ENTITY1_VERTEX_COUNT];
	groundV_Collection[0].position = DirectX::XMFLOAT4(-20.0f, -3.0f, 0.0f, 1.0f);
	groundV_Collection[0].color = DirectX::XMFLOAT4(0.0f, 0.4f, 0.3f, 1.0f);
	groundV_Collection[1].position = DirectX::XMFLOAT4(-20.0f, -3.0f, 40.0f, 1.0f);
	groundV_Collection[1].color = DirectX::XMFLOAT4(0.0f, 0.4f, 0.3f, 1.0f);
	groundV_Collection[2].position = DirectX::XMFLOAT4(20.0f, -3.0f, 0.0f, 1.0f);
	groundV_Collection[2].color = DirectX::XMFLOAT4(0.0f, 0.4f, 0.3f, 1.0f);
	groundV_Collection[3].position = DirectX::XMFLOAT4(20.0f, -3.0f, 40.0f, 1.0f); 
	groundV_Collection[3].color = DirectX::XMFLOAT4(0.0f, 0.4f, 0.3f, 1.0f);

	unsigned int* groundI_Collection = new unsigned int[ENTITY1_INDEX_COUNT];
	groundI_Collection[0] = 0;
	groundI_Collection[1] = 1;
	groundI_Collection[2] = 2;
	groundI_Collection[3] = 3;

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

	SAFE_DELETE_ARRAY(cubeV_collection);
	SAFE_DELETE_ARRAY(cubeI_collection);
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
