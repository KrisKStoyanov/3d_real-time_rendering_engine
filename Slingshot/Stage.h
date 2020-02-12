#pragma once
#include "Entity.h"

class Stage {
public:
	static Stage* Create(Entity* entityCollection, unsigned int entityCount);
	void Shutdown();
	Entity* GetEntityCollection();
	unsigned int GetEntityCount();
private:
	Stage(Entity* entityCollection, unsigned int entityCount) : 
		m_entityCollection(entityCollection), m_entityCount(entityCount) {}
	Entity* m_entityCollection;
	unsigned int m_entityCount;
};