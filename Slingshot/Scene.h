#pragma once
#include "Entity.h"

struct SCENE_DESC
{
	Entity* entityCollection;
	unsigned int entityCount;
	unsigned int mainCameraId;
};

class Scene
{
public:
	inline static Scene* Create(unsigned int id, SCENE_DESC& scene_desc)
	{
		return new Scene(id, scene_desc);
	}

	void OnUpdate();
	void Shutdown();

	inline unsigned int GetID() 
	{ 
		return m_id; 
	}

	inline void UpdateCamera(unsigned int cameraId)
	{
		(m_pEntityCollection + cameraId)->GetCamera()->Update(*(m_pEntityCollection + cameraId)->GetTransform());
	}

	inline Entity* GetCamera(unsigned int cameraId)
	{
		return (m_pEntityCollection + cameraId);
	}

	inline Entity* GetEntity(int arrayIndex)
	{
		return (m_pEntityCollection + arrayIndex);
	}

	inline Entity* GetEntityCollection()
	{
		return m_pEntityCollection;
	}

	inline unsigned int GetEntityCount()
	{
		return m_entityCount;
	}

	inline unsigned int GetMainCameraID()
	{
		return m_mainCameraId;
	}
private:
	Scene(unsigned int id, SCENE_DESC& scene_desc);

	unsigned int m_id;
	unsigned int m_mainCameraId;

	Entity* m_pEntityCollection;
	unsigned int m_entityCount;

	unsigned int m_lightStartId;
	unsigned int m_lightCount;
};

