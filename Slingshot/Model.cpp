#include "Model.h"

Model::Model(std::vector<Vertex> vData)
{
	m_VertexData = vData;
}

void Model::Setup(bool rIndexed, bool sOutput, PrimitiveType pType)
{
	renderIndexed = rIndexed;
	streamOutput = sOutput;
	primitiveType = pType;
}

void Model::Clear()
{
	m_VertexData.clear();
}
