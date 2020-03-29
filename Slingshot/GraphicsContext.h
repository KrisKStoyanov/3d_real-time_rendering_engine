#pragma once
#include "Window.h"
#include "DirectXMath.h"
#include "Buffer.h"
#include "PipelineState.h"

enum class ContextType : unsigned int
{
	D3D11 = 0,
	D3D12,
	OpenGL,
	Vulkan
};

//virtual abstract class for extending functionality across different graphics APIs through polymorphism
class GraphicsContext
{
public:
	virtual Buffer* CreateVertexBuffer(VERTEX_BUFFER_DESC) = 0;
	virtual Buffer* CreateIndexBuffer(INDEX_BUFFER_DESC) = 0;
	virtual Buffer* CreateConstantBuffer(CONSTANT_BUFFER_DESC) = 0;
};