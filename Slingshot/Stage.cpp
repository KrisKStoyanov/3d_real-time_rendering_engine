#include "Stage.h"

Stage* Stage::Create(unsigned int id, STAGE_DESC& stage_desc)
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

Entity* Stage::GetEntity(int arrayIndex)
{
	return m_pEntityCollection+arrayIndex;
}

Entity* Stage::GetEntityCollection()
{
	return m_pEntityCollection;
}

unsigned int Stage::GetEntityCount()
{
	return m_entityCount;
}

unsigned int Stage::GetMainCameraID()
{
	return m_mainCameraId;
}

unsigned int Stage::GetStartRenderID()
{
	return m_startRenderId;
}

Stage::Stage(unsigned int id, STAGE_DESC& stage_desc) :
	m_id(id), m_entityCount(stage_desc.entityCount), 
	m_startRenderId(stage_desc.startRenderId), 
	m_mainCameraId(stage_desc.mainCameraId)
{
	m_pEntityCollection = new Entity[m_entityCount];
	memcpy(m_pEntityCollection, stage_desc.entityCollection, sizeof(Entity) * m_entityCount);
	SAFE_DELETE_ARRAY(stage_desc.entityCollection);
}

