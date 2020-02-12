#include "Engine.h"

bool Engine::Initialize(WINDOW_DESC* window_desc, RENDERER_DESC* renderer_desc)
{
	if ((m_pWindow = Window::Create(window_desc)) != nullptr) {
		HWND hWnd = m_pWindow->GetHandle();
		m_isRunning = ((m_pCore = Core::Create(hWnd)) != nullptr);
		if (m_isRunning) {
			m_isRunning = m_pCore->InitializeRenderer(renderer_desc);
		}
	}

	if (m_isRunning) 
	{
		Entity* testEntity = new Entity();

		ColorShaderVertex* vertexCollection = new ColorShaderVertex[3];
		vertexCollection[0].position = DirectX::XMFLOAT4(0.0f, 0.5f, 0.5f, 1.0f);
		vertexCollection[0].color = DirectX::XMFLOAT4(1.0f, 0.0f, 0.0f, 1.0f);
		vertexCollection[1].position = DirectX::XMFLOAT4(0.5f, -0.5f, 0.5f, 1.0f);
		vertexCollection[1].color = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f);
		vertexCollection[2].position = DirectX::XMFLOAT4(-0.5f, -0.5f, 0.5f, 1.0f);
		vertexCollection[2].color = DirectX::XMFLOAT4(0.0f, 0.0f, 1.0f, 1.0f);

		unsigned int* indexCollection = new unsigned int[3];
		indexCollection[0] = 0;
		indexCollection[1] = 1;
		indexCollection[2] = 2;

		char* ColorVS_bytecode = nullptr, * ColorPS_bytecode = nullptr;
		size_t ColorVS_size, ColorPS_size;
		ColorVS_bytecode = GetFileBytecode("ColorVertexShader.cso", ColorVS_size);
		ColorPS_bytecode = GetFileBytecode("ColorPixelShader.cso", ColorPS_size);

		m_isRunning = testEntity->SetModel(m_pCore->GetRenderer(), 
			&MODEL_DESC(
				&MESH_DESC(
					VertexType::ColorShaderVertex,
					D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
					vertexCollection, 3, 
					indexCollection, 3),
				&SHADER_DESC(
					ColorVS_bytecode, ColorVS_size, 
					ColorPS_bytecode, ColorPS_size)
			)
		);

		m_isRunning = ((m_pStage = Stage::Create(testEntity, 1)) != nullptr);
	}

	return m_isRunning;
}

int Engine::Run()
{
	MSG msg = {};
	while (m_isRunning) {
		while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessageW(&msg);
			if (msg.message == WM_QUIT) {
				m_isRunning = false;
			}
		}
		m_pCore->OnUpdate(m_pStage);
	}
	return (int)msg.wParam;
}

void Engine::Shutdown()
{
	SAFE_SHUTDOWN(m_pCore);
	SAFE_SHUTDOWN(m_pWindow);
}
