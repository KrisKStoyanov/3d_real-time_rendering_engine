#pragma once
#include "Transform.h"
#include "Macros.h"
#include "Fragment.h"
#include <random>
#include <chrono>

struct Photon
{
	float x, y, z;
	float r, g, b;
	DirectX::XMVECTOR pos;
	float power;
	DirectX::XMVECTOR incDir;
	short flag;
};

struct GEOMETRY_DESC
{
	Fragment* fragmentCollection;
	int fragmentCount;
};

class PhotonMap
{
public:
	static PhotonMap* Create(GEOMETRY_DESC& desc);
	void Destroy();
	void ComputePhotons(
		DirectX::XMVECTOR& lightPos,
		float lightPower,
		int photonTargetCount);

	inline Photon* GetPhotonCollection() 
	{
		return m_photonCollection;
	}
	inline int GetPhotonCount()
	{
		return m_photonCount;
	}
private:
	PhotonMap(GEOMETRY_DESC& desc) : 
		m_fragmentCollection(desc.fragmentCollection),
		m_fragmentCount(desc.fragmentCount),
		m_photonCollection(nullptr),
		m_photonCount(0) {}
	Photon TracePhoton(
		DirectX::XMVECTOR origin, 
		DirectX::XMVECTOR dir,
		float power);

	Fragment* m_fragmentCollection;
	int m_fragmentCount;

	Photon* m_photonCollection;
	int m_photonCount;
};
