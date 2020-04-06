#pragma once
#include "Helpers.h"
#include "Macros.h"
#include "FileParsing.h"
#include "Vertex.h"

struct PIPELINE_DESC
{
	ShadingModel shadingModel;
	const char* CS_filename;
	const char* VS_filename;
	const char* HS_filename;
	const char* DS_filename;
	const char* GS_filename;
	const char* PS_filename;

	const char* VS_filename_DI;
	const char* VS_filename_DM;

	const char* PS_filename_DI;
	const char* PS_filename_DM;
};

class PipelineState 
{
public:
	virtual ShadingModel GetShadingModel()
	{
		return m_shadingModel;
	}
	virtual void SetShadingModel(ShadingModel shadingModel)
	{
		m_shadingModel = shadingModel;
	}

	//virtual void UpdatePerFrame() = 0;
	//virtual void UpdatePerModel() = 0;
	//virtual void Shutdown() = 0;

private:

	ShadingModel m_shadingModel;
};