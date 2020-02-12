#include "Stage.h"

Stage* Stage::Create(Entity* entityCollection, unsigned int entityCount)
{
	return new Stage(entityCollection, entityCount);
}

void Stage::Shutdown()
{

}

Entity* Stage::GetEntityCollection()
{
	return m_entityCollection;
}

unsigned int Stage::GetEntityCount()
{
	return m_entityCount;
}

