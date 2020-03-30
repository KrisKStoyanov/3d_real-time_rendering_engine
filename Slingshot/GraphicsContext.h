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
	virtual bool Initialize() = 0;
	virtual void Shutdown() = 0;

	//virtual void BindPipelineState(PipelineState&) = 0;
	//virtual void BindMeshBuffers(Buffer& vertexBuffer, Buffer& indexBuffer) = 0;
	//virtual void BindConstantBuffer(Buffer& constantBuffer) = 0;

	virtual void StartFrameRender() = 0;
	virtual void EndFrameRender() = 0;

	virtual void DrawIndexed(
		unsigned int indexCount,
		unsigned int startIndex,
		unsigned int baseVertexLocation) = 0;
	//virtual void Draw() = 0;
	//virtual void DrawInstanced() = 0;

	virtual PipelineState* CreatePipelineState(PIPELINE_DESC) = 0;
	virtual Buffer* CreateVertexBuffer(VERTEX_BUFFER_DESC) = 0;
	virtual Buffer* CreateIndexBuffer(INDEX_BUFFER_DESC) = 0;
	virtual Buffer* CreateConstantBuffer(CONSTANT_BUFFER_DESC) = 0;
};