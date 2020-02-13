#pragma once
#include "Entity.h"

class Stage {
public:
	static Stage* Create(
		Camera* cameraCollection, 
		unsigned int cameraCount,
		unsigned int mainCameraId,
		Entity* entityCollection, 
		unsigned int entityCount);
	void Shutdown();

	void SetMainCamera(unsigned int mainCameraId);
	Camera* GetMainCamera();

	Entity* GetEntityCollection();
	unsigned int GetEntityCount();
private:
	Stage(
		Camera* cameraCollection,
		unsigned int cameraCount,
		unsigned int mainCameraId,
		Entity* entityCollection,
		unsigned int entityCount) :
		m_cameraCollection(cameraCollection),
		m_cameraCount(cameraCount),
		m_mainCameraId(mainCameraId),
		m_entityCollection(entityCollection), 
		m_entityCount(entityCount) 
	{}

	Camera* m_cameraCollection;
	unsigned int m_cameraCount;

	unsigned int m_mainCameraId;

	Entity* m_entityCollection;
	unsigned int m_entityCount;
};