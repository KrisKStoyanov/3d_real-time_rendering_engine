#pragma once
#include "Entity.h"

struct STAGE_DESC {
	Entity* entityCollection;
	unsigned int entityCount;
	unsigned int mainCameraId;
	STAGE_DESC(
		Entity* _entityCollection,
		unsigned int _entityCount,
		unsigned int _mainCameraId) :
		entityCollection(_entityCollection),
		entityCount(_entityCount),
		mainCameraId(_mainCameraId)
	{}
};

class Stage {
public:
	static Stage* Create(unsigned int id, STAGE_DESC* stage_desc);
	void Shutdown();
	unsigned int GetID();

	Entity* GetMainCamera();

	Entity* GetEntityCollection();
	unsigned int GetEntityCount();
private:
	Stage(unsigned int id, STAGE_DESC* stage_desc) :
		m_id(id),
		m_pEntityCollection(stage_desc->entityCollection),
		m_entityCount(stage_desc->entityCount),
		m_mainCameraId(stage_desc->mainCameraId)
	{}

	unsigned int m_id;

	Entity* m_pEntityCollection;
	unsigned int m_entityCount;
	unsigned int m_mainCameraId;
};