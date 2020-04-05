#pragma once
#include "DirectXMath.h"
#include "Material.h"
#include <math.h>

struct INTERACTION_DESC
{
	DirectX::XMVECTOR pos;
	DirectX::XMVECTOR normal;
	DirectX::XMVECTOR incDir;
	float incPhotonPower;
};

class Fragment
{
public:
	virtual bool Intersect(
		DirectX::XMVECTOR origin,
		DirectX::XMVECTOR dir,
		INTERACTION_DESC& desc) = 0;
};

class Sphere : public Fragment
{
public:
	Sphere(DirectX::XMVECTOR pos, SPHERE_DESC& sphere_desc, MATERIAL_DESC& mat_desc);
	virtual bool Intersect(
		DirectX::XMVECTOR origin, 
		DirectX::XMVECTOR dir, 
		INTERACTION_DESC& desc) override;

	float m_radius;
	DirectX::XMVECTOR m_pos;
	Material* m_pMaterial;
};

class Plane : public Fragment
{
public:
	Plane(DirectX::XMVECTOR pos, PLANE_DESC& plane_desc, MATERIAL_DESC& mat_desc);
	virtual bool Intersect(
		DirectX::XMVECTOR origin, 
		DirectX::XMVECTOR dir,
		INTERACTION_DESC& desc) override;

	DirectX::XMVECTOR m_normal;
	DirectX::XMVECTOR m_pos;
	Material* m_pMaterial;
};

class Triangle : public Fragment
{
public:
	Triangle(DirectX::XMVECTOR pos, TRIANGLE_DESC& tri_desc, MATERIAL_DESC& mat_desc);
	virtual bool Intersect(
		DirectX::XMVECTOR origin, 
		DirectX::XMVECTOR dir,
		INTERACTION_DESC& desc) override;

	DirectX::XMVECTOR m_point0, m_point1, m_point2;
	DirectX::XMVECTOR m_normal0, m_normal1, m_normal2;

	DirectX::XMVECTOR m_pos;
	Material* m_pMaterial;
};

class Cube : public Fragment
{
public:
	Cube(DirectX::XMVECTOR pos, CUBE_DESC cube_desc, MATERIAL_DESC& mat_desc);
	virtual bool Intersect(
		DirectX::XMVECTOR origin,
		DirectX::XMVECTOR dir,
		INTERACTION_DESC& desc) override;

	DirectX::XMVECTOR m_pos;
	Plane* m_pSides[6];
};