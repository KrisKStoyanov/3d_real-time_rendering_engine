#include "Stage.h"

Stage* Stage::Create(unsigned int id, STAGE_DESC* stage_desc)
{
	return new Stage(id, stage_desc);
}

void Stage::Shutdown()
{
	for (unsigned int i = 0; i < m_entityCount; ++i) {
		(m_pEntityCollection + i)->Shutdown();
	}
}

unsigned int Stage::GetID()
{
	return m_id;
}

Entity* Stage::GetMainCamera()
{
	return (m_pEntityCollection+m_mainCameraId);
}

Entity* Stage::GetEntityCollection()
{
	return m_pEntityCollection;
}

unsigned int Stage::GetEntityCount()
{
	return m_entityCount;
}

