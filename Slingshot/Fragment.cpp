#include "Fragment.h"

Sphere::Sphere(DirectX::XMVECTOR pos, SPHERE_DESC& sphere_desc, MATERIAL_DESC& mat_desc) : 
	m_pos(pos)
{
	m_pMaterial = Material::Create(mat_desc);
}

bool Sphere::Intersect(
	DirectX::XMVECTOR origin, 
	DirectX::XMVECTOR dir, 
	INTERACTION_DESC& desc)
{
	//Relative position to light source
	DirectX::XMVECTOR posToSource = DirectX::XMVectorSubtract(m_pos, origin);

	DirectX::XMVECTOR vRes = DirectX::XMVector4Dot(posToSource, dir);
	float dirCoefficient = DirectX::XMVectorGetX(vRes);
	if (dirCoefficient < 0)
	{
		return false;
	}

	vRes = DirectX::XMVector4Dot(posToSource, posToSource);
	float distToRightAngleSqrd = DirectX::XMVectorGetX(vRes) - (dirCoefficient * dirCoefficient);
	float radiusSqrd = m_radius * m_radius;
	if (distToRightAngleSqrd > radiusSqrd) 
	{
		return false;
	}

	float distFromRightAngleToIntPoint = sqrtf(radiusSqrd - distToRightAngleSqrd);
	float distToIntPos = dirCoefficient - distFromRightAngleToIntPoint;
	float distToExit = dirCoefficient + distFromRightAngleToIntPoint;

	DirectX::XMVECTOR distToIntPosV = DirectX::XMVectorSet(distToIntPos, distToIntPos, distToIntPos, distToIntPos);
	DirectX::XMVECTOR intPos = DirectX::XMVectorAdd(origin, DirectX::XMVectorMultiply(distToIntPosV, dir));
	DirectX::XMVECTOR intNormal = DirectX::XMVector4Normalize(DirectX::XMVectorSubtract(intPos, m_pos));

	desc.pos = intPos;
	desc.normal = intNormal;
	return true;
}

Plane::Plane(DirectX::XMVECTOR pos, PLANE_DESC& plane_desc, MATERIAL_DESC& mat_desc) : 
	m_pos(pos)
{
	m_pMaterial = Material::Create(mat_desc);
}

bool Plane::Intersect(
	DirectX::XMVECTOR origin, 
	DirectX::XMVECTOR dir, 
	INTERACTION_DESC& desc)
{
	DirectX::XMVECTOR denomV = DirectX::XMVector4Dot(dir, m_normal);
	float denom = DirectX::XMVectorGetX(denomV);

	DirectX::XMVECTOR distToIntPosV = DirectX::XMVectorDivide(
		DirectX::XMVector4Dot(DirectX::XMVectorSubtract(m_pos, origin), m_normal),
		denomV);
	float distToIntPos = DirectX::XMVectorGetX(distToIntPosV);

	if (distToIntPos <= 0) 
	{
		return false;
	}

	return false;
}

Triangle::Triangle(DirectX::XMVECTOR pos, TRIANGLE_DESC& tri_desc, MATERIAL_DESC& mat_desc) : 
	m_pos(pos)
{
	m_pMaterial = Material::Create(mat_desc);
}

bool Triangle::Intersect(
	DirectX::XMVECTOR origin, 
	DirectX::XMVECTOR dir, 
	INTERACTION_DESC& desc)
{
	DirectX::XMVECTOR edge0 = DirectX::XMVectorSubtract(m_point1, m_point0);
	DirectX::XMVECTOR edge1 = DirectX::XMVectorSubtract(m_point2, m_point0);

	DirectX::XMVECTOR uV = DirectX::XMVectorDivide(
		DirectX::XMVector4Dot(DirectX::XMVectorSubtract(origin, m_point0), DirectX::XMVector3Cross(dir, edge1)), 
		DirectX::XMVector4Dot(edge0, DirectX::XMVector3Cross(dir, edge1)));

	float u = DirectX::XMVectorGetX(uV);

	DirectX::XMVECTOR vV = DirectX::XMVectorDivide(
		DirectX::XMVector4Dot(dir, DirectX::XMVector3Cross(DirectX::XMVectorSubtract(origin, m_point0), edge0)),
		DirectX::XMVector4Dot(edge0, DirectX::XMVector3Cross(dir, edge1)));

	float v = DirectX::XMVectorGetX(vV);

	if (u < 0 || u > 1) {
		return false;
	}
	else if (v < 0 || (u + v) > 1) {
		return false;
	}

	DirectX::XMVECTOR distToSourceV = DirectX::XMVectorDivide(
		DirectX::XMVector4Dot(edge1, DirectX::XMVector3Cross(DirectX::XMVectorSubtract(origin, m_point0), edge0)),
		DirectX::XMVector4Dot(edge0, DirectX::XMVector3Cross(dir, edge1)));

	DirectX::XMVECTOR intPos = DirectX::XMVectorAdd(origin, DirectX::XMVectorMultiply(dir, distToSourceV));
	float distToSource = DirectX::XMVectorGetX(distToSourceV);
	float w = 1 - u - v;
	DirectX::XMVECTOR wV = DirectX::XMVectorSet(w, w, w, w);

	DirectX::XMVECTOR intNormal =
		DirectX::XMVector4Normalize(
			DirectX::XMVectorAdd(
				DirectX::XMVectorAdd(
					DirectX::XMVectorMultiply(wV, m_normal0),
					DirectX::XMVectorMultiply(uV, m_normal1)),
					DirectX::XMVectorMultiply(vV, m_normal2)));

	desc.pos = intPos;
	desc.normal = intNormal;
	return true;
}

Cube::Cube(DirectX::XMVECTOR pos, CUBE_DESC cube_desc, MATERIAL_DESC& mat_desc) : 
	m_pos(pos)
{
	//Front
	PLANE_DESC plane_desc0;
	plane_desc0.width = cube_desc.width;
	plane_desc0.length = cube_desc.height;

	PLANE_DESC plane_desc1;
	plane_desc1.width = cube_desc.width;
	plane_desc1.length = cube_desc.height;

	PLANE_DESC plane_desc2;
	plane_desc2.width = cube_desc.width;
	plane_desc2.length = cube_desc.height;

	PLANE_DESC plane_desc3;
	plane_desc3.width = cube_desc.width;
	plane_desc3.length = cube_desc.height;

	PLANE_DESC plane_desc4;
	plane_desc4.width = cube_desc.width;
	plane_desc4.length = cube_desc.height;

	PLANE_DESC plane_desc5;
	plane_desc5.width = cube_desc.width;
	plane_desc5.length = cube_desc.height;
}

bool Cube::Intersect(DirectX::XMVECTOR origin, DirectX::XMVECTOR dir, INTERACTION_DESC& desc)
{
	//find closest plane/edge to origin and check dot product to find angle/intersect point
	return false;
}
