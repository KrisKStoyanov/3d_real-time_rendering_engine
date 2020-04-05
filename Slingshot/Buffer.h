#pragma once
#include "Vertex.h"
#include "Macros.h"
#include "Helpers.h"

struct VERTEX_BUFFER_DESC
{
	Vertex* vertexCollection;
	int vertexCount;
	Topology topology;
	int stride;
	int offset;
};

struct INDEX_BUFFER_DESC
{
	int* indexCollection;
	int indexCount;
};

struct CONSTANT_BUFFER_DESC
{
	void* cbufferData;
	int cbufferSize;
	ShaderType shaderType;
	int registerSlot;
};

struct STRUCTURED_BUFFER_DESC
{
	int structureSize;
	int byteStride; //<- size of largest struct element
	int registerSlot;
	int numStructs;
	int numElementsPerStruct;
};

class Buffer
{
public:
	virtual unsigned int GetElementCount() = 0;
	virtual void Destroy() = 0;
private:

};

