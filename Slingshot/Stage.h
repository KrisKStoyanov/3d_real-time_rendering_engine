#pragma once
#include "Entity.h"

struct STAGE_DESC {
	Entity* entityCollection;
	unsigned int entityCount;
	unsigned int mainCameraId;
	unsigned int startRenderId;
};

class Stage {
public:
	static Stage* Create(unsigned int id, STAGE_DESC& stage_desc);
	void Shutdown();
	unsigned int GetID();

	inline void UpdateCamera(unsigned int cameraId)
	{
		(m_pEntityCollection + cameraId)->GetCamera()->Update(*(m_pEntityCollection + cameraId)->GetTransform());
	}

	Entity* GetMainCamera();

	Entity* GetEntity(int arrayIndex);

	Entity* GetEntityCollection();
	unsigned int GetEntityCount();
	unsigned int GetMainCameraID();
	unsigned int GetStartRenderID();
private:
	Stage(unsigned int id, STAGE_DESC& stage_desc);

	unsigned int m_id;

	Entity* m_pEntityCollection;
	unsigned int m_entityCount;

	unsigned int m_mainCameraId;
	unsigned int m_startRenderId;
};