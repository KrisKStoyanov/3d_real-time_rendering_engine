#pragma once
#include "Entity.h"

struct SCENE_DESC
{
	Entity* entityCollection;
	int entityCount;
	int mainCameraId;
};

class Scene
{
public:
	inline static Scene* Create(int id, SCENE_DESC& scene_desc)
	{
		return new Scene(id, scene_desc);
	}

	void OnUpdate();
	void Shutdown();

	inline int GetID() 
	{ 
		return m_id; 
	}

	inline void UpdateCamera(int cameraId)
	{
		(m_pEntityCollection + cameraId)->GetCamera()->Update(*(m_pEntityCollection + cameraId)->GetTransform());
	}

	inline Entity* GetCamera(int cameraId)
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

	inline int GetEntityCount()
	{
		return m_entityCount;
	}

	inline int GetMainCameraID()
	{
		return m_mainCameraId;
	}
private:
	Scene(int id, SCENE_DESC& scene_desc);

	int m_id;
	int m_mainCameraId;

	Entity* m_pEntityCollection;
	int m_entityCount;

	int m_lightStartId;
	int m_lightCount;
};

