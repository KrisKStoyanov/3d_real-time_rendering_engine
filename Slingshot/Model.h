#pragma once
#include <vector>
#include "Vertex.h"

enum class PrimitiveType : unsigned int {
	Triangle = 0,
	Point,
	Line,
};

class Model
{
public:
	Model(std::vector<Vertex> vData);

	void Setup(
		bool rIndexed = false, 
		bool sOutput = false, 
		PrimitiveType pType = PrimitiveType::Triangle);
	void Clear();

	std::vector<Vertex> m_VertexData;

	bool renderIndexed = false;
	bool streamOutput = false;
	PrimitiveType primitiveType = PrimitiveType::Triangle;

};

