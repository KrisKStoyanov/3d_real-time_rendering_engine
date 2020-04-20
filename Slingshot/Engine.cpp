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
#if defined(_DEBUG)
	AllocConsole();
#endif

	if ((m_pWindow = Window::Create(window_desc)) != nullptr) 
	{
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
		PIPELINE_DESC pipeline_desc;
		pipeline_desc.shadingModel = ShadingModel::FinalGathering;
		pipeline_desc.VS_filename_DI = "DirectIlluminationVS.cso";
		pipeline_desc.PS_filename_DI = "DirectIlluminationPS.cso";
		pipeline_desc.VS_filename_DM = "DepthMapVS.cso";
		pipeline_desc.PS_filename_DM = "DepthMapPS.cso";
		m_isRunning = m_pRenderer->Initialize(pipeline_desc);	
		GEOMETRY_DESC geo_desc = SetupScene(*m_pScene, 11);
		//m_pRenderer->SetupPhotonMap(geo_desc);
		m_pTimer = new Timer();
	}

	return m_isRunning;
}

GEOMETRY_DESC Engine::SetupScene(Scene& scene, const int entityCount)
{
	Entity* entityCollection = new Entity[entityCount];

	//	MAIN CAMERA
	//------------------------------
	RECT winRect;
	GetWindowRect(m_pWindow->GetHandle(), &winRect);
	float resX = static_cast<float>(winRect.right - winRect.left);
	float resY = static_cast<float>(winRect.bottom - winRect.top);
	CAMERA_DESC mainCameraDesc;
	mainCameraDesc.lenseWidth = resX;
	mainCameraDesc.lenseHeight = resY;
	entityCollection[0].SetCamera(mainCameraDesc);
	TRANSFORM_DESC entity0_transform_desc;
	entity0_transform_desc.position = DirectX::XMFLOAT4(0.0f, 5.0f, -15.0f, 1.0f);
	entityCollection[0].SetTransform(entity0_transform_desc);
	//------------------------------

	//	LIGHTING
	//------------------------------
	LIGHT_DESC entity1_light_desc;
	entityCollection[1].SetLight(entity1_light_desc);
	TRANSFORM_DESC entity1_transform_desc;
	entity1_transform_desc.position = DirectX::XMFLOAT4(0.0f, 5.0f, -10.0f, 1.0f);//DirectX::XMFLOAT4(0.0f, 12.5f, 10.0f, 1.0f); //DirectX::XMFLOAT4(0.0f, 5.0f, -10.0f, 1.0f);
	//entity1_transform_desc.rotation = DirectX::XMFLOAT4(90.0f, 0.0f, 0.0f, 0.0f);
	entityCollection[1].SetTransform(entity1_transform_desc);
	//------------------------------

	//	CORNEL BOX
	//------------------------------
	//Bottom
	TRANSFORM_DESC entity2_transform_desc;
	entity2_transform_desc.position = DirectX::XMFLOAT4(0.0f, -5.0f, 10.0f, 1.0f);
	entityCollection[2].SetTransform(entity2_transform_desc);
	MATERIAL_DESC mat_desc0;
	mat_desc0.shadingModel = ShadingModel::FinalGathering;
	mat_desc0.surfaceColor = DirectX::XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
	mat_desc0.roughness = 1.0f;
	PLANE_DESC plane_desc0;
	plane_desc0.length = 10.0f;
	plane_desc0.width = 10.0f;
	CreatePlane(*(entityCollection + 2), plane_desc0, mat_desc0);

	//Front
	TRANSFORM_DESC entity3_transform_desc;
	entity3_transform_desc.position = DirectX::XMFLOAT4(0.0f, 5.0f, 20.0f, 1.0f);
	entity3_transform_desc.rotation = DirectX::XMFLOAT4(-90.0f, 0.0f, 0.0f, 0.0f);
	entityCollection[3].SetTransform(entity3_transform_desc);
	MATERIAL_DESC mat_desc1;
	mat_desc1.shadingModel = ShadingModel::FinalGathering;
	mat_desc1.surfaceColor = DirectX::XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
	mat_desc1.roughness = 1.0f;
	PLANE_DESC plane_desc1;
	plane_desc1.length = 10.0f;
	plane_desc1.width = 10.0f;
	CreatePlane(*(entityCollection + 3), plane_desc1, mat_desc1);

	//Left
	TRANSFORM_DESC entity4_transform_desc;
	entity4_transform_desc.position = DirectX::XMFLOAT4(-10.0f, 5.0f, 10.0f, 1.0f);
	entity4_transform_desc.rotation = DirectX::XMFLOAT4(0.0f, 0.0f, -90.0f, 0.0f);
	entityCollection[4].SetTransform(entity4_transform_desc);
	MATERIAL_DESC mat_desc2;
	mat_desc2.shadingModel = ShadingModel::FinalGathering;
	mat_desc2.surfaceColor = DirectX::XMFLOAT4(1.0f, 0.0f, 0.0f, 1.0f);
	mat_desc2.roughness = 1.0f;
	PLANE_DESC plane_desc2;
	plane_desc2.length = 10.0f;
	plane_desc2.width = 10.0f;
	CreatePlane(*(entityCollection + 4), plane_desc2, mat_desc2);

	//Right
	TRANSFORM_DESC entity5_transform_desc;
	entity5_transform_desc.position = DirectX::XMFLOAT4(10.0f, 5.0f, 10.0f, 1.0f);
	entity5_transform_desc.rotation = DirectX::XMFLOAT4(0.0f, 0.0f, 90.0f, 0.0f);
	entityCollection[5].SetTransform(entity5_transform_desc);
	MATERIAL_DESC mat_desc3;
	mat_desc3.shadingModel = ShadingModel::FinalGathering;
	mat_desc3.surfaceColor = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f);
	mat_desc3.roughness = 1.0f;
	PLANE_DESC plane_desc3;
	plane_desc3.length = 10.0f;
	plane_desc3.width = 10.0f;
	CreatePlane(*(entityCollection + 5), plane_desc3, mat_desc3);

	//Top
	TRANSFORM_DESC entity6_transform_desc;
	entity6_transform_desc.position = DirectX::XMFLOAT4(0.0f, 15.0f, 10.0f, 1.0f);
	entity6_transform_desc.rotation = DirectX::XMFLOAT4(-180.0f, 0.0f, 0.0f, 0.0f);
	entityCollection[6].SetTransform(entity6_transform_desc);
	MATERIAL_DESC mat_desc4;
	mat_desc4.shadingModel = ShadingModel::FinalGathering;
	mat_desc4.surfaceColor = DirectX::XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
	mat_desc4.roughness = 1.0f;
	PLANE_DESC plane_desc4;
	plane_desc4.length = 10.0f;
	plane_desc4.width = 10.0f;
	CreatePlane(*(entityCollection + 6), plane_desc4, mat_desc4);

	//Back
	TRANSFORM_DESC entity7_transform_desc;
	entity7_transform_desc.position = DirectX::XMFLOAT4(0.0f, 5.0f, 0.0f, 1.0f);
	entity7_transform_desc.rotation = DirectX::XMFLOAT4(90.0f, 0.0f, 0.0f, 0.0f);
	entityCollection[7].SetTransform(entity7_transform_desc);
	MATERIAL_DESC mat_desc5;
	mat_desc5.shadingModel = ShadingModel::FinalGathering;
	mat_desc5.surfaceColor = DirectX::XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
	mat_desc5.roughness = 1.0f;
	PLANE_DESC plane_desc5;
	plane_desc5.length = 10.0f;
	plane_desc5.width = 10.0f;
	CreatePlane(*(entityCollection + 7), plane_desc5, mat_desc5);
	//------------------------------

	//	DECOR
	//------------------------------
	//Yellow ball
	TRANSFORM_DESC entity8_transform_desc;
	entity8_transform_desc.position = DirectX::XMFLOAT4(0.0f, -1.5f, 6.0f, 1.0f);
	entityCollection[8].SetTransform(entity8_transform_desc);
	MATERIAL_DESC mat_desc6;
	mat_desc6.shadingModel = ShadingModel::FinalGathering;
	mat_desc6.surfaceColor = DirectX::XMFLOAT4(1.0f, 1.0f, 0.0f, 1.0f);
	mat_desc6.roughness = 1.0f;
	SPHERE_DESC sphere_desc0;
	sphere_desc0.slices = 50;
	sphere_desc0.stacks = 50;
	sphere_desc0.radius = 3.5f;
	CreateSphere(*(entityCollection + 8), sphere_desc0, mat_desc6);

	//Teal ball
	TRANSFORM_DESC entity9_transform_desc;
	entity9_transform_desc.position = DirectX::XMFLOAT4(5.0f, 0.0f, 13.5f, 1.0f);
	entityCollection[9].SetTransform(entity9_transform_desc);
	MATERIAL_DESC mat_desc7;
	mat_desc7.shadingModel = ShadingModel::FinalGathering;
	mat_desc7.surfaceColor = DirectX::XMFLOAT4(0.0f, 1.0f, 1.0f, 1.0f);
	mat_desc7.roughness = 1.0f;
	SPHERE_DESC sphere_desc1;
	sphere_desc1.slices = 50;
	sphere_desc1.stacks = 50;
	sphere_desc1.radius = 5.0f;
	CreateSphere(*(entityCollection + 9), sphere_desc1, mat_desc7);
	//------------------------------

	//Purple Cube 
	TRANSFORM_DESC entity10_transform_desc;
	entity10_transform_desc.position = DirectX::XMFLOAT4(-5.0f, 0.0f, 13.5f, 1.0f);
	entity10_transform_desc.rotation = DirectX::XMFLOAT4(0.0f, 12.5f, 0.0f, 0.0f);
	entityCollection[10].SetTransform(entity10_transform_desc);
	MATERIAL_DESC mat_desc8;
	mat_desc8.shadingModel = ShadingModel::FinalGathering;
	mat_desc8.surfaceColor = DirectX::XMFLOAT4(1.0f, 0.0f, 1.0f, 1.0f);
	mat_desc8.roughness = 1.0f;
	CUBE_DESC cube_desc0;
	cube_desc0.width = 3.25f;
	cube_desc0.height = 5.0f;
	cube_desc0.length = 3.25f;
	CreateCube(*(entityCollection + 10), cube_desc0, mat_desc8);
	//------------------------------

	SCENE_DESC scene_desc;
	scene_desc.entityCollection = new Entity[entityCount];
	memcpy(scene_desc.entityCollection, entityCollection, sizeof(Entity) * entityCount);
	scene_desc.entityCount = entityCount;
	scene_desc.mainCameraId = 0;
	scene_desc.startLightId = 1;
	scene_desc.lightCount = 1;
	m_pScene = Scene::Create(0, scene_desc);

	SAFE_DELETE_ARRAY(entityCollection);

	GEOMETRY_DESC geo_desc{};
	geo_desc.fragmentCollection;
	geo_desc.fragmentCollection[0] = Plane(
		DirectX::XMVectorSet(
			entity2_transform_desc.position.x, 
			entity2_transform_desc.position.y, 
			entity2_transform_desc.position.z, 
			entity2_transform_desc.position.w), 
		plane_desc0, mat_desc0);
	geo_desc.fragmentCollection[1] = Plane(
		DirectX::XMVectorSet(
			entity3_transform_desc.position.x,
			entity3_transform_desc.position.y,
			entity3_transform_desc.position.z,
			entity3_transform_desc.position.w),
		plane_desc1, mat_desc1);
	geo_desc.fragmentCollection[2] = Plane(
		DirectX::XMVectorSet(
			entity4_transform_desc.position.x,
			entity4_transform_desc.position.y,
			entity4_transform_desc.position.z,
			entity4_transform_desc.position.w),
		plane_desc2, mat_desc2);
	geo_desc.fragmentCollection[3] = Plane(
		DirectX::XMVectorSet(
			entity5_transform_desc.position.x,
			entity5_transform_desc.position.y,
			entity5_transform_desc.position.z,
			entity5_transform_desc.position.w),
		plane_desc3, mat_desc3);
	geo_desc.fragmentCollection[4] = Plane(
		DirectX::XMVectorSet(
			entity6_transform_desc.position.x,
			entity6_transform_desc.position.y,
			entity6_transform_desc.position.z,
			entity6_transform_desc.position.w),
		plane_desc4, mat_desc4);

	geo_desc.fragmentCollection[5] = Sphere(
		DirectX::XMVectorSet(
			entity7_transform_desc.position.x,
			entity7_transform_desc.position.y,
			entity7_transform_desc.position.z,
			entity7_transform_desc.position.w),
		sphere_desc0, mat_desc5);
	geo_desc.fragmentCollection[6] = Sphere(
		DirectX::XMVectorSet(
			entity8_transform_desc.position.x,
			entity8_transform_desc.position.y,
			entity8_transform_desc.position.z,
			entity8_transform_desc.position.w),
		sphere_desc1, mat_desc6);
	geo_desc.fragmentCollection[7] = Cube(
		DirectX::XMVectorSet(
			entity9_transform_desc.position.x,
			entity9_transform_desc.position.y,
			entity9_transform_desc.position.z,
			entity9_transform_desc.position.w),
		cube_desc0, mat_desc7);

	geo_desc.fragmentCount = 8;
	return geo_desc;
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
		m_pScene->OnUpdate();
		m_pRenderer->Draw(*m_pScene);
	}
	Shutdown();
	return static_cast<int>(msg.wParam);
}

void Engine::Shutdown()
{
	SAFE_SHUTDOWN(m_pScene);
	SAFE_SHUTDOWN(m_pRenderer);
	SAFE_SHUTDOWN(m_pWindow);
	SAFE_DELETE(m_pTimer);
}

void Engine::CreatePlane(Entity& entity, PLANE_DESC& plane_desc, MATERIAL_DESC& material_desc)
{
	//Plane Object
	//------------------------------
	const int VERTEX_COUNT = 4;
	const int INDEX_COUNT = 4;

	MESH_DESC meshDesc;

	switch (material_desc.shadingModel)
	{
	case ShadingModel::GoochShading:
	{
		GoochShadingVertex* entityV_collection = new GoochShadingVertex[VERTEX_COUNT];

		entityV_collection[0].position = DirectX::XMFLOAT4(-1.0f * plane_desc.width, 0.0f, -1.0f * plane_desc.length, 1.0f);
		entityV_collection[0].normal = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 0.0f);

		entityV_collection[1].position = DirectX::XMFLOAT4(-1.0f * plane_desc.width, 0.0f, 1.0f * plane_desc.length, 1.0f);
		entityV_collection[1].normal = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 0.0f);

		entityV_collection[2].position = DirectX::XMFLOAT4(1.0f * plane_desc.width, 0.0f, -1.0f * plane_desc.length, 1.0f);
		entityV_collection[2].normal = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 0.0f);

		entityV_collection[3].position = DirectX::XMFLOAT4(1.0f * plane_desc.width, 0.0f, 1.0f * plane_desc.length, 1.0f);
		entityV_collection[3].normal = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 0.0f);

		int* entityI_collection = new int[INDEX_COUNT];
		entityI_collection[0] = 0;
		entityI_collection[1] = 1;
		entityI_collection[2] = 2;
		entityI_collection[3] = 3;

		meshDesc.vertex_buffer_desc.vertexCollection = new GoochShadingVertex[VERTEX_COUNT];
		memcpy(meshDesc.vertex_buffer_desc.vertexCollection, entityV_collection, sizeof(GoochShadingVertex) * VERTEX_COUNT);
		meshDesc.vertex_buffer_desc.vertexCount = VERTEX_COUNT;
		meshDesc.vertex_buffer_desc.topology = Topology::TRIANGLESTRIP;
		
		meshDesc.index_buffer_desc.indexCollection = new int[INDEX_COUNT];
		memcpy(meshDesc.index_buffer_desc.indexCollection, entityI_collection, sizeof(int) * INDEX_COUNT);
		meshDesc.index_buffer_desc.indexCount = INDEX_COUNT;

		SAFE_DELETE_ARRAY(entityV_collection);
		SAFE_DELETE_ARRAY(entityI_collection);
	}
	break;
	case ShadingModel::OrenNayarShading:
	{
		OrenNayarVertex* entityV_collection = new OrenNayarVertex[VERTEX_COUNT];

		entityV_collection[0].position = DirectX::XMFLOAT4(-1.0f * plane_desc.width, 0.0f, -1.0f * plane_desc.length, 1.0f);
		entityV_collection[0].normal = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 0.0f);

		entityV_collection[1].position = DirectX::XMFLOAT4(-1.0f * plane_desc.width, 0.0f, 1.0f * plane_desc.length, 1.0f);
		entityV_collection[1].normal = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 0.0f);

		entityV_collection[2].position = DirectX::XMFLOAT4(1.0f * plane_desc.width, 0.0f, -1.0f * plane_desc.length, 1.0f);
		entityV_collection[2].normal = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 0.0f);

		entityV_collection[3].position = DirectX::XMFLOAT4(1.0f * plane_desc.width, 0.0f, 1.0f * plane_desc.length, 1.0f);
		entityV_collection[3].normal = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 0.0f);

		int* entityI_collection = new int[INDEX_COUNT];
		entityI_collection[0] = 0;
		entityI_collection[1] = 1;
		entityI_collection[2] = 2;
		entityI_collection[3] = 3;

		meshDesc.vertex_buffer_desc.vertexCollection = new OrenNayarVertex[VERTEX_COUNT];
		memcpy(meshDesc.vertex_buffer_desc.vertexCollection, entityV_collection, sizeof(OrenNayarVertex) * VERTEX_COUNT);
		meshDesc.vertex_buffer_desc.vertexCount = VERTEX_COUNT;
		meshDesc.vertex_buffer_desc.topology = Topology::TRIANGLESTRIP;

		meshDesc.index_buffer_desc.indexCollection = new int[INDEX_COUNT];
		memcpy(meshDesc.index_buffer_desc.indexCollection, entityI_collection, sizeof(int) * INDEX_COUNT);
		meshDesc.index_buffer_desc.indexCount = INDEX_COUNT;

		SAFE_DELETE_ARRAY(entityV_collection);
		SAFE_DELETE_ARRAY(entityI_collection);
	}
	break;
	case ShadingModel::FinalGathering:
	{
		FinalGatheringVertex* entityV_collection = new FinalGatheringVertex[VERTEX_COUNT];

		entityV_collection[0].position = DirectX::XMFLOAT4(-1.0f * plane_desc.width, 0.0f, -1.0f * plane_desc.length, 1.0f);
		entityV_collection[0].normal = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 0.0f);
		entityV_collection[0].uv = DirectX::XMFLOAT2(0.0f, 0.0f);

		entityV_collection[1].position = DirectX::XMFLOAT4(-1.0f * plane_desc.width, 0.0f, 1.0f * plane_desc.length, 1.0f);
		entityV_collection[1].normal = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 0.0f);
		entityV_collection[1].uv = DirectX::XMFLOAT2(0.0f, 1.0f);

		entityV_collection[2].position = DirectX::XMFLOAT4(1.0f * plane_desc.width, 0.0f, -1.0f * plane_desc.length, 1.0f);
		entityV_collection[2].normal = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 0.0f);
		entityV_collection[2].uv = DirectX::XMFLOAT2(1.0f, 0.0f);

		entityV_collection[3].position = DirectX::XMFLOAT4(1.0f * plane_desc.width, 0.0f, 1.0f * plane_desc.length, 1.0f);
		entityV_collection[3].normal = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 0.0f);
		entityV_collection[3].uv = DirectX::XMFLOAT2(1.0f, 1.0f);

		int* entityI_collection = new int[INDEX_COUNT];
		entityI_collection[0] = 0;
		entityI_collection[1] = 1;
		entityI_collection[2] = 2;
		entityI_collection[3] = 3;

		meshDesc.vertex_buffer_desc.vertexCollection = new FinalGatheringVertex[VERTEX_COUNT];
		memcpy(meshDesc.vertex_buffer_desc.vertexCollection, entityV_collection, sizeof(FinalGatheringVertex) * VERTEX_COUNT);
		meshDesc.vertex_buffer_desc.vertexCount = VERTEX_COUNT;
		meshDesc.vertex_buffer_desc.topology = Topology::TRIANGLESTRIP;

		meshDesc.index_buffer_desc.indexCollection = new int[INDEX_COUNT];
		memcpy(meshDesc.index_buffer_desc.indexCollection, entityI_collection, sizeof(int) * INDEX_COUNT);
		meshDesc.index_buffer_desc.indexCount = INDEX_COUNT;

		SAFE_DELETE_ARRAY(entityV_collection);
		SAFE_DELETE_ARRAY(entityI_collection);
	}
	break;
	}

	entity.SetModel(
		*m_pRenderer->GetGraphicsContext(), meshDesc, material_desc);
}

void Engine::CreateCube(Entity& entity, CUBE_DESC& cube_desc, MATERIAL_DESC& material_desc)
{
	//Cube Object - Interpolated Normals
	//------------------------------
	const int VERTEX_COUNT = 8;
	const int INDEX_COUNT = 14;

	MESH_DESC meshDesc;

	switch (material_desc.shadingModel)
	{
	case ShadingModel::GoochShading:
	{
		GoochShadingVertex* entityV_collection = new GoochShadingVertex[VERTEX_COUNT];
		entityV_collection[0].position = DirectX::XMFLOAT4(-1.0f * cube_desc.width, -1.0f * cube_desc.height, -1.0f * cube_desc.length, 1.0f);
		entityV_collection[0].normal = DirectX::XMFLOAT4(-1.0f, -1.0f, -1.0f, 0.0f);

		entityV_collection[1].position = DirectX::XMFLOAT4(-1.0f * cube_desc.width, 1.0f * cube_desc.height, -1.0f * cube_desc.length, 1.0f);
		entityV_collection[1].normal = DirectX::XMFLOAT4(-1.0f, 1.0f, -1.0f, 0.0f);

		entityV_collection[2].position = DirectX::XMFLOAT4(1.0f * cube_desc.width, -1.0f * cube_desc.height, -1.0f * cube_desc.length, 1.0f);
		entityV_collection[2].normal = DirectX::XMFLOAT4(1.0f, -1.0f, -1.0f, 0.0f);

		entityV_collection[3].position = DirectX::XMFLOAT4(1.0f * cube_desc.width, 1.0f * cube_desc.height, -1.0f * cube_desc.length, 1.0f);
		entityV_collection[3].normal = DirectX::XMFLOAT4(1.0f, 1.0f, -1.0f, 0.0f);

		entityV_collection[4].position = DirectX::XMFLOAT4(-1.0f * cube_desc.width, -1.0f * cube_desc.height, 1.0f * cube_desc.length, 1.0f);
		entityV_collection[4].normal = DirectX::XMFLOAT4(-1.0f, -1.0f, 1.0f, 0.0f);

		entityV_collection[5].position = DirectX::XMFLOAT4(-1.0f * cube_desc.width, 1.0f * cube_desc.height, 1.0f * cube_desc.length, 1.0f);
		entityV_collection[5].normal = DirectX::XMFLOAT4(-1.0f, 1.0f, 1.0f, 0.0f);

		entityV_collection[6].position = DirectX::XMFLOAT4(1.0f * cube_desc.width, -1.0f * cube_desc.height, 1.0f * cube_desc.length, 1.0f);
		entityV_collection[6].normal = DirectX::XMFLOAT4(1.0f, -1.0f, 1.0f, 0.0f);

		entityV_collection[7].position = DirectX::XMFLOAT4(1.0f * cube_desc.width, 1.0f * cube_desc.height, 1.0f * cube_desc.length, 1.0f);
		entityV_collection[7].normal = DirectX::XMFLOAT4(1.0f, 1.0f, 1.0f, 0.0f);

		int* entityI_collection = new int[INDEX_COUNT];

		//Front
		entityI_collection[0] = 0;
		entityI_collection[1] = 1;
		entityI_collection[2] = 2;
		entityI_collection[3] = 3;

		entityI_collection[4] = 7;
		entityI_collection[5] = 1;

		entityI_collection[6] = 5;
		entityI_collection[7] = 0;

		entityI_collection[8] = 4;
		entityI_collection[9] = 2;

		entityI_collection[10] = 6;
		entityI_collection[11] = 7;

		entityI_collection[12] = 4;
		entityI_collection[13] = 5;

		meshDesc.vertex_buffer_desc.vertexCollection = new GoochShadingVertex[VERTEX_COUNT];
		memcpy(meshDesc.vertex_buffer_desc.vertexCollection, entityV_collection, sizeof(GoochShadingVertex) * VERTEX_COUNT);
		meshDesc.vertex_buffer_desc.vertexCount = VERTEX_COUNT;
		meshDesc.vertex_buffer_desc.topology = Topology::TRIANGLESTRIP;

		meshDesc.index_buffer_desc.indexCollection = new int[INDEX_COUNT];
		memcpy(meshDesc.index_buffer_desc.indexCollection, entityI_collection, sizeof(int) * INDEX_COUNT);
		meshDesc.index_buffer_desc.indexCount = INDEX_COUNT;

		SAFE_DELETE_ARRAY(entityV_collection);
		SAFE_DELETE_ARRAY(entityI_collection);
	}
	break;
	case ShadingModel::OrenNayarShading:
	{
		OrenNayarVertex* entityV_collection = new OrenNayarVertex[VERTEX_COUNT];
		entityV_collection[0].position = DirectX::XMFLOAT4(-1.0f * cube_desc.width, -1.0f * cube_desc.height, -1.0f * cube_desc.length, 1.0f);
		entityV_collection[0].normal = DirectX::XMFLOAT4(-1.0f, -1.0f, -1.0f, 0.0f);

		entityV_collection[1].position = DirectX::XMFLOAT4(-1.0f * cube_desc.width, 1.0f * cube_desc.height, -1.0f * cube_desc.length, 1.0f);
		entityV_collection[1].normal = DirectX::XMFLOAT4(-1.0f, 1.0f, -1.0f, 0.0f);

		entityV_collection[2].position = DirectX::XMFLOAT4(1.0f * cube_desc.width, -1.0f * cube_desc.height, -1.0f * cube_desc.length, 1.0f);
		entityV_collection[2].normal = DirectX::XMFLOAT4(1.0f, -1.0f, -1.0f, 0.0f);

		entityV_collection[3].position = DirectX::XMFLOAT4(1.0f * cube_desc.width, 1.0f * cube_desc.height, -1.0f * cube_desc.length, 1.0f);
		entityV_collection[3].normal = DirectX::XMFLOAT4(1.0f, 1.0f, -1.0f, 0.0f);

		entityV_collection[4].position = DirectX::XMFLOAT4(-1.0f * cube_desc.width, -1.0f * cube_desc.height, 1.0f * cube_desc.length, 1.0f);
		entityV_collection[4].normal = DirectX::XMFLOAT4(-1.0f, -1.0f, 1.0f, 0.0f);

		entityV_collection[5].position = DirectX::XMFLOAT4(-1.0f * cube_desc.width, 1.0f * cube_desc.height, 1.0f * cube_desc.length, 1.0f);
		entityV_collection[5].normal = DirectX::XMFLOAT4(-1.0f, 1.0f, 1.0f, 0.0f);

		entityV_collection[6].position = DirectX::XMFLOAT4(1.0f * cube_desc.width, -1.0f * cube_desc.height, 1.0f * cube_desc.length, 1.0f);
		entityV_collection[6].normal = DirectX::XMFLOAT4(1.0f, -1.0f, 1.0f, 0.0f);

		entityV_collection[7].position = DirectX::XMFLOAT4(1.0f * cube_desc.width, 1.0f * cube_desc.height, 1.0f * cube_desc.length, 1.0f);
		entityV_collection[7].normal = DirectX::XMFLOAT4(1.0f, 1.0f, 1.0f, 0.0f);

		int* entityI_collection = new int[INDEX_COUNT];

		//Front
		entityI_collection[0] = 0;
		entityI_collection[1] = 1;
		entityI_collection[2] = 2;
		entityI_collection[3] = 3;

		entityI_collection[4] = 7;
		entityI_collection[5] = 1;

		entityI_collection[6] = 5;
		entityI_collection[7] = 0;

		entityI_collection[8] = 4;
		entityI_collection[9] = 2;

		entityI_collection[10] = 6;
		entityI_collection[11] = 7;

		entityI_collection[12] = 4;
		entityI_collection[13] = 5;

		meshDesc.vertex_buffer_desc.vertexCollection = new OrenNayarVertex[VERTEX_COUNT];
		memcpy(meshDesc.vertex_buffer_desc.vertexCollection, entityV_collection, sizeof(OrenNayarVertex)* VERTEX_COUNT);
		meshDesc.vertex_buffer_desc.vertexCount = VERTEX_COUNT;
		meshDesc.vertex_buffer_desc.topology = Topology::TRIANGLESTRIP;

		meshDesc.index_buffer_desc.indexCollection = new int[INDEX_COUNT];
		memcpy(meshDesc.index_buffer_desc.indexCollection, entityI_collection, sizeof(int)* INDEX_COUNT);
		meshDesc.index_buffer_desc.indexCount = INDEX_COUNT;

		SAFE_DELETE_ARRAY(entityV_collection);
		SAFE_DELETE_ARRAY(entityI_collection);
	}
	break;
	case ShadingModel::FinalGathering:
	{
		FinalGatheringVertex* entityV_collection = new FinalGatheringVertex[VERTEX_COUNT];
		entityV_collection[0].position = DirectX::XMFLOAT4(-1.0f * cube_desc.width, -1.0f * cube_desc.height, -1.0f * cube_desc.length, 1.0f);
		entityV_collection[0].normal = DirectX::XMFLOAT4(-1.0f, -1.0f, -1.0f, 0.0f);
		entityV_collection[0].uv = DirectX::XMFLOAT2(0.0f, 0.0f);

		entityV_collection[1].position = DirectX::XMFLOAT4(-1.0f * cube_desc.width, 1.0f * cube_desc.height, -1.0f * cube_desc.length, 1.0f);
		entityV_collection[1].normal = DirectX::XMFLOAT4(-1.0f, 1.0f, -1.0f, 0.0f);
		entityV_collection[1].uv = DirectX::XMFLOAT2(0.0f, 1.0f);

		entityV_collection[2].position = DirectX::XMFLOAT4(1.0f * cube_desc.width, -1.0f * cube_desc.height, -1.0f * cube_desc.length, 1.0f);
		entityV_collection[2].normal = DirectX::XMFLOAT4(1.0f, -1.0f, -1.0f, 0.0f);
		entityV_collection[2].uv = DirectX::XMFLOAT2(1.0f, 0.0f);

		entityV_collection[3].position = DirectX::XMFLOAT4(1.0f * cube_desc.width, 1.0f * cube_desc.height, -1.0f * cube_desc.length, 1.0f);
		entityV_collection[3].normal = DirectX::XMFLOAT4(1.0f, 1.0f, -1.0f, 0.0f);
		entityV_collection[3].uv = DirectX::XMFLOAT2(1.0f, 1.0f);

		entityV_collection[4].position = DirectX::XMFLOAT4(-1.0f * cube_desc.width, -1.0f * cube_desc.height, 1.0f * cube_desc.length, 1.0f);
		entityV_collection[4].normal = DirectX::XMFLOAT4(-1.0f, -1.0f, 1.0f, 0.0f);
		entityV_collection[4].uv = DirectX::XMFLOAT2(0.0f, 0.0f);

		entityV_collection[5].position = DirectX::XMFLOAT4(-1.0f * cube_desc.width, 1.0f * cube_desc.height, 1.0f * cube_desc.length, 1.0f);
		entityV_collection[5].normal = DirectX::XMFLOAT4(-1.0f, 1.0f, 1.0f, 0.0f);
		entityV_collection[5].uv = DirectX::XMFLOAT2(0.0f, 1.0f);

		entityV_collection[6].position = DirectX::XMFLOAT4(1.0f * cube_desc.width, -1.0f * cube_desc.height, 1.0f * cube_desc.length, 1.0f);
		entityV_collection[6].normal = DirectX::XMFLOAT4(1.0f, -1.0f, 1.0f, 0.0f);
		entityV_collection[6].uv = DirectX::XMFLOAT2(1.0f, 0.0f);

		entityV_collection[7].position = DirectX::XMFLOAT4(1.0f * cube_desc.width, 1.0f * cube_desc.height, 1.0f * cube_desc.length, 1.0f);
		entityV_collection[7].normal = DirectX::XMFLOAT4(1.0f, 1.0f, 1.0f, 0.0f);
		entityV_collection[7].uv = DirectX::XMFLOAT2(1.0f, 1.0f);

		int* entityI_collection = new int[INDEX_COUNT];

		//Front
		entityI_collection[0] = 0;
		entityI_collection[1] = 1;
		entityI_collection[2] = 2;
		entityI_collection[3] = 3;

		entityI_collection[4] = 7;
		entityI_collection[5] = 1;

		entityI_collection[6] = 5;
		entityI_collection[7] = 0;

		entityI_collection[8] = 4;
		entityI_collection[9] = 2;

		entityI_collection[10] = 6;
		entityI_collection[11] = 7;

		entityI_collection[12] = 4;
		entityI_collection[13] = 5;

		meshDesc.vertex_buffer_desc.vertexCollection = new FinalGatheringVertex[VERTEX_COUNT] ;
		memcpy(meshDesc.vertex_buffer_desc.vertexCollection, entityV_collection, sizeof(FinalGatheringVertex)* VERTEX_COUNT);
		meshDesc.vertex_buffer_desc.vertexCount = VERTEX_COUNT;
		meshDesc.vertex_buffer_desc.topology = Topology::TRIANGLESTRIP;

		meshDesc.index_buffer_desc.indexCollection = new int[INDEX_COUNT];
		memcpy(meshDesc.index_buffer_desc.indexCollection, entityI_collection, sizeof(int)* INDEX_COUNT);
		meshDesc.index_buffer_desc.indexCount = INDEX_COUNT;

		SAFE_DELETE_ARRAY(entityV_collection);
		SAFE_DELETE_ARRAY(entityI_collection);
	}
	break;
	}

	entity.SetModel(
		*m_pRenderer->GetGraphicsContext(), meshDesc, material_desc);
}

void Engine::CreateSphere(Entity& entity, SPHERE_DESC& sphere_desc, MATERIAL_DESC& material_desc) 
{
	const int VERTEX_COUNT = (sphere_desc.stacks + 1) * (sphere_desc.slices + 1);
	const int INDEX_COUNT = (sphere_desc.slices * sphere_desc.stacks + sphere_desc.slices) * 6;

	MESH_DESC meshDesc;

	int* entityI_collection = new int[INDEX_COUNT];

	float slicesF = static_cast<float>(sphere_desc.slices);
	float stacksF = static_cast<float>(sphere_desc.stacks);

	switch (material_desc.shadingModel)
	{
	case ShadingModel::GoochShading:
	{
		GoochShadingVertex* entityV_collection = new GoochShadingVertex[VERTEX_COUNT];

		for (int i = 0; i < sphere_desc.stacks + 1; ++i)
		{
			float V = i / stacksF;
			float phi = V * 3.14f;

			for (int j = 0; j < sphere_desc.slices + 1; ++j) {

				float U = j / slicesF;
				float theta = U * 6.28f;

				float sinPhi = sinf(phi);

				float z = cosf(theta) * sinPhi; //reverse x/z formula to fit drawing order
				float y = cosf(phi);
				float x = sinf(theta) * sinPhi;

				int index = j + i * (sphere_desc.slices + 1);
				entityV_collection[index].position = DirectX::XMFLOAT4(x * sphere_desc.radius, y * sphere_desc.radius, z * sphere_desc.radius, 1.0f);
				entityV_collection[index].normal = DirectX::XMFLOAT4(x, y, z, 0.0f);
			}
		}

		meshDesc.vertex_buffer_desc.vertexCollection = new GoochShadingVertex[VERTEX_COUNT];
		memcpy(meshDesc.vertex_buffer_desc.vertexCollection, entityV_collection, sizeof(GoochShadingVertex) * VERTEX_COUNT);
		meshDesc.vertex_buffer_desc.vertexCount = VERTEX_COUNT;
		meshDesc.vertex_buffer_desc.topology = Topology::TRIANGLESTRIP;

		SAFE_DELETE_ARRAY(entityV_collection);
	}
	break;
	case ShadingModel::OrenNayarShading:
	{
		OrenNayarVertex* entityV_collection = new OrenNayarVertex[VERTEX_COUNT];

		for (int i = 0; i < sphere_desc.stacks + 1; ++i)
		{
			float V = i / stacksF;
			float phi = V * 3.14f;

			for (int j = 0; j < sphere_desc.slices + 1; ++j) {

				float U = j / slicesF;
				float theta = U * 6.28f;

				float sinPhi = sinf(phi);

				float z = cosf(theta) * sinPhi; //reverse x/z formula to fit drawing order
				float y = cosf(phi);
				float x = sinf(theta) * sinPhi;

				int index = j + i * (sphere_desc.slices + 1);
				entityV_collection[index].position = DirectX::XMFLOAT4(x * sphere_desc.radius, y * sphere_desc.radius, z * sphere_desc.radius, 1.0f);
				entityV_collection[index].normal = DirectX::XMFLOAT4(x, y, z, 0.0f);
			}
		}

		meshDesc.vertex_buffer_desc.vertexCollection = new OrenNayarVertex[VERTEX_COUNT];
		memcpy(meshDesc.vertex_buffer_desc.vertexCollection, entityV_collection, sizeof(OrenNayarVertex) * VERTEX_COUNT);
		meshDesc.vertex_buffer_desc.vertexCount = VERTEX_COUNT;
		meshDesc.vertex_buffer_desc.topology = Topology::TRIANGLESTRIP;

		SAFE_DELETE_ARRAY(entityV_collection);
	}
	break;
	case ShadingModel::FinalGathering:
	{
		FinalGatheringVertex* entityV_collection = new FinalGatheringVertex[VERTEX_COUNT];

		for (int i = 0; i < sphere_desc.stacks + 1; ++i)
		{
			float V = i / stacksF;
			float phi = V * 3.14f;

			for (int j = 0; j < sphere_desc.slices + 1; ++j) {

				float U = j / slicesF;
				float theta = U * 6.28f;

				float sinPhi = sinf(phi);

				float z = cosf(theta) * sinPhi; //reverse x/z formula to fit drawing order
				float y = cosf(phi);
				float x = sinf(theta) * sinPhi;

				int index = j + i * (sphere_desc.slices + 1);
				entityV_collection[index].position = DirectX::XMFLOAT4(x * sphere_desc.radius, y * sphere_desc.radius, z * sphere_desc.radius, 1.0f);
				entityV_collection[index].normal = DirectX::XMFLOAT4(x, y, z, 0.0f);
				entityV_collection[index].uv = DirectX::XMFLOAT2((asin(entityV_collection[index].normal.x) / DirectX::XM_PI + 0.5f), (asin(entityV_collection[index].normal.y) / DirectX::XM_PI + 0.5f));
			}
		}

		meshDesc.vertex_buffer_desc.vertexCollection = new FinalGatheringVertex[VERTEX_COUNT];
		memcpy(meshDesc.vertex_buffer_desc.vertexCollection, entityV_collection, sizeof(FinalGatheringVertex) * VERTEX_COUNT);
		meshDesc.vertex_buffer_desc.vertexCount = VERTEX_COUNT;
		meshDesc.vertex_buffer_desc.topology = Topology::TRIANGLESTRIP;

		SAFE_DELETE_ARRAY(entityV_collection);
	}
	break;
	}
	
	//entityV_collection[0].position = DirectX::XMFLOAT4(0.0f, sphere_desc.radius, 0.0f, 1.0f);
	//entityV_collection[0].normal = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 0.0f);
	//entityV_collection[0].uv = DirectX::XMFLOAT2(0.0f, 0.0f);

	//float phiStep = DirectX::XM_PI / sphere_desc.stacks;
	//float thetaStep = DirectX::XM_2PI / sphere_desc.slices;

	//for (int i = 1; i <= sphere_desc.stacks - 1; ++i)
	//{
	//	float phi = i * phiStep;
	//	for (int j = 0; j <= sphere_desc.slices; ++j) 
	//	{
	//		float theta = j * thetaStep;

	//		float z = sinf(phi) * cosf(theta); //reverse x/z position to fit drawing order
	//		float y = cosf(phi);
	//		float x = sinf(phi) * sinf(theta);

	//		int index = j + i * (sphere_desc.slices);
	//		entityV_collection[index].position = DirectX::XMFLOAT4(x * sphere_desc.radius, y * sphere_desc.radius, z * sphere_desc.radius, 1.0f);
	//		entityV_collection[index].normal = DirectX::XMFLOAT4(x, y, z, 0.0f);
	//		entityV_collection[index].uv = DirectX::XMFLOAT2(theta / DirectX::XM_2PI, phi / DirectX::XM_PI);
	//	}
	//}
	//entityV_collection[VERTEX_COUNT - 1].position = DirectX::XMFLOAT4(0.0f, -sphere_desc.radius, 0.0f, 1.0f);
	//entityV_collection[VERTEX_COUNT - 1].normal = DirectX::XMFLOAT4(0.0f, -1.0f, 0.0f, 0.0f);
	//entityV_collection[VERTEX_COUNT - 1].uv = DirectX::XMFLOAT2(0.0f, 1.0f);

	int index = 0;
	for (int i = 0; i < sphere_desc.slices * sphere_desc.stacks + sphere_desc.slices; ++i)
	{
		entityI_collection[index] = i;
		entityI_collection[index + 1] = i + sphere_desc.slices + 1;
		entityI_collection[index + 2] = i + sphere_desc.slices;
		entityI_collection[index + 3] = i + sphere_desc.slices + 1;
		entityI_collection[index + 4] = i;
		entityI_collection[index + 5] = i + 1;

		index += 6;
	}

	meshDesc.index_buffer_desc.indexCollection = new int[INDEX_COUNT];
	memcpy(meshDesc.index_buffer_desc.indexCollection, entityI_collection, sizeof(int) * INDEX_COUNT);
	meshDesc.index_buffer_desc.indexCount = INDEX_COUNT;

	SAFE_DELETE_ARRAY(entityI_collection);

	entity.SetModel(
		*m_pRenderer->GetGraphicsContext(), meshDesc, material_desc);
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
		//SetRect(&rcClip, ptMin.x, ptMin.y, ptMax.x, ptMax.y);
		//ClipCursor(&rcClip);
		ShowCursor(false);
		int xOffset = (ptMax.x - ptMin.x);
		int yOffset = (ptMax.y - ptMin.y);
		int xCoord = ptMax.x - xOffset / 2;
		int yCoord = ptMax.y - yOffset / 2;
		SetCursorPos(xCoord, yCoord);
		m_pScene->GetCamera(m_pScene->GetMainCameraID())->GetCamera()->SetMouseCoord(xCoord, yCoord);
		m_pScene->GetCamera(m_pScene->GetMainCameraID())->GetCamera()->SetRotateStatus(true);
	}
	break;
	case WM_LBUTTONUP:
	{
		ReleaseCapture();
		ShowCursor(true);
		m_pScene->GetCamera(m_pScene->GetMainCameraID())->GetCamera()->SetRotateStatus(false);
		return DefWindowProc(hWnd, uMsg, wParam, lParam);
	}
	break;
	case WM_MOUSEMOVE:
	{
		if (m_pScene->GetCamera(m_pScene->GetMainCameraID())->GetCamera()->GetRotateStatus()) {
			int xCoord = GET_X_LPARAM(lParam);
			int yCoord = GET_Y_LPARAM(lParam);
			POINT pt = { xCoord, yCoord };
			ClientToScreen(hWnd, &pt);
			int lastMouseX, lastMouseY;
			m_pScene->GetCamera(m_pScene->GetMainCameraID())->GetCamera()->GetMouseCoord(lastMouseX, lastMouseY);
			m_pScene->GetCamera(m_pScene->GetMainCameraID())->GetCamera()->SetMouseCoord(pt.x, pt.y);
			float offsetX = static_cast<float>(pt.x) - lastMouseX;
			float offsetY = static_cast<float>(pt.y) - lastMouseY;
			float pitch = offsetX * m_pScene->GetCamera(m_pScene->GetMainCameraID())->GetCamera()->GetRotationSpeed();
			float head = offsetY * m_pScene->GetCamera(m_pScene->GetMainCameraID())->GetCamera()->GetRotationSpeed();
			m_pScene->GetCamera(m_pScene->GetMainCameraID())->GetTransform()->RotateEulerAngles(head, pitch, 0.0f);
		}
	}
	break;
	case WM_KEYDOWN:
	{
		short repeatCode = *((short*)&lParam + 1);
		using namespace DirectX;
		switch (wParam) {
		case 0x57: //W 
		{
			if (repeatCode == 16401)
			{
				m_pScene->GetCamera(m_pScene->GetMainCameraID())->GetTransform()->Translate(
					m_pScene->GetCamera(m_pScene->GetMainCameraID())->GetCamera()->GetTranslationSpeed() *
					m_pScene->GetCamera(m_pScene->GetMainCameraID())->GetTransform()->GetForwardDir() *
					m_pTimer->m_smoothstepF);
			}
		}
		break;
		case 0x41: //A
		{
			if (repeatCode == 16414)
			{
				m_pScene->GetCamera(m_pScene->GetMainCameraID())->GetTransform()->Translate(
					m_pScene->GetCamera(m_pScene->GetMainCameraID())->GetCamera()->GetTranslationSpeed() * -1.0f *
					m_pScene->GetCamera(m_pScene->GetMainCameraID())->GetTransform()->GetRightDir() *
					m_pTimer->m_smoothstepF);
			}
		}
		break;
		case 0x53: //S
		{
			if (repeatCode == 16415) 
			{
				m_pScene->GetCamera(m_pScene->GetMainCameraID())->GetTransform()->Translate(
					m_pScene->GetCamera(m_pScene->GetMainCameraID())->GetCamera()->GetTranslationSpeed() * -1.0f *
					m_pScene->GetCamera(m_pScene->GetMainCameraID())->GetTransform()->GetForwardDir() *
					m_pTimer->m_smoothstepF);
			}
		}
		break;
		case 0x44: //D
		{
			if (repeatCode == 16416) 
			{
				m_pScene->GetCamera(m_pScene->GetMainCameraID())->GetTransform()->Translate(
					m_pScene->GetCamera(m_pScene->GetMainCameraID())->GetCamera()->GetTranslationSpeed() *
					m_pScene->GetCamera(m_pScene->GetMainCameraID())->GetTransform()->GetRightDir() *
					m_pTimer->m_smoothstepF);
			}
		}
		break;
		case VK_F1:
		{
			m_pRenderer->UpdateConstantVRS();
			return DefWindowProc(hWnd, uMsg, wParam, lParam);
		}
		break;
		case VK_F2:
		{
			m_pRenderer->ToggleConstantVRS();
			return DefWindowProc(hWnd, uMsg, wParam, lParam);
		}
		break;
		case VK_ESCAPE:
		{
			DestroyWindow(hWnd);
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
		m_pScene->GetCamera(m_pScene->GetMainCameraID())->GetCamera()->SetRotateStatus(false);
		return DefWindowProc(hWnd, uMsg, wParam, lParam);
	}
	break;
	case WM_QUIT:
	case WM_DESTROY:
	{
#if defined(_DEBUG)
		FreeConsole();
#endif
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