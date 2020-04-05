#include "PhotonMap.h"

PhotonMap* PhotonMap::Create(GEOMETRY_DESC& desc)
{
	return new PhotonMap(desc);
}

void PhotonMap::Destroy()
{
	SAFE_DELETE_ARRAY(m_photonCollection);
}

void PhotonMap::ComputePhotons(DirectX::XMVECTOR& lightPos, float lightPower, int photonTargetCount)
{
	m_photonCount = photonTargetCount;
	m_photonCollection = new Photon[m_photonCount];

    std::mt19937_64 rng;
    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{ uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> 32) };
    rng.seed(ss);
    std::uniform_real_distribution<float> unif(-1.0f, 1.0f);

	int photonsEmittedCount = 0;
	float photonPower = lightPower / photonTargetCount;
	srand(static_cast<unsigned>(time(0)));
	while (photonsEmittedCount < m_photonCount)
	{
		float photonXDir = unif(rng);
		float photonYDir = unif(rng);
		float photonZDir = unif(rng);
		if (!(photonXDir * photonXDir + photonYDir * photonYDir + photonZDir * photonZDir) > 1.0f)
		{
			DirectX::XMVECTOR photonDir = DirectX::XMVectorSet(photonXDir, photonYDir, photonZDir, 0.0f);
			m_photonCollection[photonsEmittedCount] = TracePhoton(lightPos, photonDir, photonPower);
			photonsEmittedCount++;
		}
	}
}

Photon PhotonMap::TracePhoton(DirectX::XMVECTOR origin, DirectX::XMVECTOR dir, float power)
{
	DirectX::XMVECTOR photonSamplePos = origin;
	float remainingPower = power;
	for (int i = 0; i < m_fragmentCount; ++i)
	{
		INTERACTION_DESC desc;
		if (m_fragmentCollection[i].Intersect(origin, dir, desc))
		{
			//decide probabilistically based on intersected material properties 
			float procedureSelector = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

			////diffuse reflection
			//if (procedureSelector < m_fragmentCollection[i].m_diffuseReflectionCoefficient)
			//{

			//}
			////specular reflection
			//else if (procedureSelector < 
			//	m_fragmentCollection[i].m_diffuseReflectionCoefficient + 
			//	m_fragmentCollection[i].m_specularReflectionCoefficient)
			//{

			//}
			////absorb
			//else
			//{
			//	photonSamplePos = desc.pos;
			//	remainingPower = power;
			//}
			//break;
		}
	}
	Photon photon;
	photon.pos = photonSamplePos;
	photon.power = remainingPower;
	return photon;
}
