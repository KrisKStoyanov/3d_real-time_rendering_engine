#include "GraphicsProps.h"

bool GraphicsProps::Setup(D3D11Context* context)
{
	size_t VS_size, PS_size;
	char* VS_bytecode = nullptr, * PS_bytecode = nullptr;

	if ((VS_bytecode = GetShaderBytecode("ColorShader_VS.cso", VS_size)) == nullptr) {
		setup = false;
	}

	if ((PS_bytecode = GetShaderBytecode("ColorShader_PS.cso", PS_size)) == nullptr) {
		setup = false;
	}

	D3D11_INPUT_ELEMENT_DESC VS_inputLayout[2];

	VS_inputLayout[0].SemanticName = "POSITION";
	VS_inputLayout[0].SemanticIndex = 0;
	VS_inputLayout[0].Format = DXGI_FORMAT::DXGI_FORMAT_R32G32B32A32_FLOAT;
	VS_inputLayout[0].InputSlot = 0;
	VS_inputLayout[0].AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
	VS_inputLayout[0].InputSlotClass = D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA;
	VS_inputLayout[0].InstanceDataStepRate = 0;

	VS_inputLayout[1].SemanticName = "COLOR";
	VS_inputLayout[1].SemanticIndex = 0;
	VS_inputLayout[1].Format = DXGI_FORMAT::DXGI_FORMAT_R32G32B32A32_FLOAT;
	VS_inputLayout[1].InputSlot = 0;
	VS_inputLayout[1].AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
	VS_inputLayout[1].InputSlotClass = D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA;
	VS_inputLayout[1].InstanceDataStepRate = 0;

	return setup;
}

void GraphicsProps::Clear()
{

}

bool GraphicsProps::GetSetupStatus()
{
	return setup;
}

char* GraphicsProps::GetShaderBytecode(const char* filename, size_t& filesize)
{
	std::ifstream shaderFileStream;
	shaderFileStream.open(filename, std::ios::in | std::ios::binary | std::ios::ate);
	if (shaderFileStream.is_open()) {
		shaderFileStream.seekg(0, std::ios::end);
		filesize = static_cast<size_t>(shaderFileStream.tellg());
		shaderFileStream.seekg(0, std::ios::beg);
		char* bytecode = new char[filesize];
		shaderFileStream.read(bytecode, filesize);
		return bytecode;
	}
	return nullptr;
}
