#pragma once
#include "Vertex.h"
#include "Macros.h"

struct VERTEX_BUFFER_DESC
{
	Vertex* vertexCollection;
	unsigned int vertexCount;
	Topology topology;
	unsigned int stride;
	unsigned int offset;
};

struct INDEX_BUFFER_DESC
{
	unsigned int* indexCollection;
	unsigned int indexCount;
};

class Buffer
{
public:

private:

};

