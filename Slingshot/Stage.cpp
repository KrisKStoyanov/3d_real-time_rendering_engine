#include "Stage.h"

Stage* Stage::Create(
	Camera* cameraCollection,
	unsigned int cameraCount,
	unsigned int mainCameraId,
	Entity* entityCollection,
	unsigned int entityCount)
{
	return new Stage(cameraCollection, cameraCount, mainCameraId, entityCollection, entityCount);
}

void Stage::Shutdown()
{
	for (unsigned int i = 0; i < m_entityCount; ++i) {
		(m_entityCollection + i)->Shutdown();
	}
}

void Stage::SetMainCamera(unsigned int mainCameraId)
{
	if (mainCameraId < m_cameraCount) {
		m_mainCameraId = mainCameraId;
	}
}

Camera* Stage::GetMainCamera()
{
	return (m_cameraCollection + m_mainCameraId);
}

Entity* Stage::GetEntityCollection()
{
	return m_entityCollection;
}

unsigned int Stage::GetEntityCount()
{
	return m_entityCount;
}

