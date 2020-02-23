#pragma once
#include "Entity.h"

struct STAGE_DESC {
	unsigned int entityCount;
	unsigned int mainCameraId;
};

class Stage {
public:
	static Stage* Create(unsigned int id, STAGE_DESC& stage_desc, Entity& entityCol);
	void Shutdown();
	unsigned int GetID();

	Entity* GetMainCamera();

	Entity* GetEntityCollection();
	unsigned int GetEntityCount();
private:
	Stage(unsigned int id, STAGE_DESC& stage_desc, Entity& entityCol);

	unsigned int m_id;

	Entity* m_pEntityCollection;
	unsigned int m_entityCount;
	unsigned int m_mainCameraId;
};