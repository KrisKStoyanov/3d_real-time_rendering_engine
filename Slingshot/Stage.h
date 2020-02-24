#pragma once
#include "Entity.h"

struct STAGE_DESC {
	Entity* entityCollection;
	unsigned int entityCount;
	unsigned int mainCameraId;
};

class Stage {
public:
	static Stage* Create(unsigned int id, STAGE_DESC& stage_desc);
	void Shutdown();
	unsigned int GetID();

	Entity* GetMainCamera();

	Entity* GetEntity(int arrayIndex);

	Entity* GetEntityCollection();
	unsigned int GetEntityCount();
	unsigned int GetMainCameraID();
private:
	Stage(unsigned int id, STAGE_DESC& stage_desc);

	unsigned int m_id;

	Entity* m_pEntityCollection;
	unsigned int m_entityCount;
	unsigned int m_mainCameraId;
};