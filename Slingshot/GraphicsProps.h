#pragma once
#include "D3D11Context.h"
#include "Vertex.h"

class GraphicsProps {
public:
	bool Setup(D3D11Context* context);
	void Clear();
	bool GetSetupStatus();
private:
	char* GetShaderBytecode(const char* filename, size_t& filesize);
	bool setup;
	Microsoft::WRL::ComPtr<ID3D11VertexShader> m_pVS;
	Microsoft::WRL::ComPtr<ID3D11PixelShader> m_pPS;
	Microsoft::WRL::ComPtr<ID3D11InputLayout> m_pIL;
};
