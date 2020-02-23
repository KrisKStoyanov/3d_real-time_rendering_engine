#include "GraphicsProps.h"

GraphicsProps* GraphicsProps::Create(D3D11Context& graphicsContext, SHADER_DESC& shader_desc, VertexType vertexType)
{
	return new GraphicsProps(graphicsContext, shader_desc, vertexType);
}

GraphicsProps::GraphicsProps(D3D11Context& graphicsContext, SHADER_DESC& shader_desc, VertexType vertexType) :
	m_pVS(nullptr), m_pPS(nullptr), m_pIL(nullptr)
{
	graphicsContext.GetDevice()->CreateVertexShader(shader_desc.VS_bytecode, shader_desc.VS_size, nullptr, &m_pVS);
	graphicsContext.GetDevice()->CreatePixelShader(shader_desc.PS_bytecode, shader_desc.PS_size, nullptr, &m_pPS);

	switch (vertexType)
	{
	case VertexType::ColorShaderVertex:
	{
		D3D11_INPUT_ELEMENT_DESC VS_inputLayout[2];

		VS_inputLayout[0].SemanticName = "POSITION";
		VS_inputLayout[0].SemanticIndex = 0;
		VS_inputLayout[0].Format = DXGI_FORMAT::DXGI_FORMAT_R32G32B32A32_FLOAT;
		VS_inputLayout[0].InputSlot = 0;
		VS_inputLayout[0].AlignedByteOffset = 0;
		VS_inputLayout[0].InputSlotClass = D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA;
		VS_inputLayout[0].InstanceDataStepRate = 0;

		VS_inputLayout[1].SemanticName = "COLOR";
		VS_inputLayout[1].SemanticIndex = 0;
		VS_inputLayout[1].Format = DXGI_FORMAT::DXGI_FORMAT_R32G32B32A32_FLOAT;
		VS_inputLayout[1].InputSlot = 0;
		VS_inputLayout[1].AlignedByteOffset = sizeof(float) * 4;
		VS_inputLayout[1].InputSlotClass = D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA;
		VS_inputLayout[1].InstanceDataStepRate = 0;
		graphicsContext.GetDevice()->CreateInputLayout(VS_inputLayout, 2, shader_desc.VS_bytecode, shader_desc.VS_size, &m_pIL);
	}
	break;
	}

	SAFE_DELETE_ARRAY(shader_desc.VS_bytecode);
	SAFE_DELETE_ARRAY(shader_desc.PS_bytecode);
}

void GraphicsProps::Shutdown()
{
	SAFE_RELEASE(m_pVS);
	SAFE_RELEASE(m_pPS);
	SAFE_RELEASE(m_pIL);
}

ID3D11VertexShader* GraphicsProps::GetVertexShader()
{
	return m_pVS;
}

ID3D11PixelShader* GraphicsProps::GetPixelShader()
{
	return m_pPS;
}

ID3D11InputLayout* GraphicsProps::GetInputLayout()
{
	return m_pIL;
}

