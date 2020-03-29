#pragma once
#include "Helpers.h"
#include "Material.h"
#include "FileParsing.h"

struct PIPELINE_DESC
{
	ShadingModel shadingModel;
	const char* VS_filename;
	const char* PS_filename;
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
private:

	ShadingModel m_shadingModel;
};