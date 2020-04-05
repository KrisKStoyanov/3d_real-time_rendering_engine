#include "Scene.h"

void Scene::OnUpdate()
{
	for (int i = 0; i < m_entityCount; ++i) 
	{
		Entity& entity = *(m_pEntityCollection + i);
		entity.GetTransform()->Update();
		if (entity.GetLight())
		{
			entity.GetTransform()->SetViewMatrix();
			entity.GetTransform()->SetProjectionMatrix(1.0f, 100.0f);
		}
	}
	UpdateCamera(m_mainCameraId);
}

void Scene::Shutdown()
{
	for (int i = 0; i < m_entityCount; ++i) 
	{
		(m_pEntityCollection + i)->Shutdown();
	}
}

Scene::Scene(int id, SCENE_DESC& scene_desc) :
	m_id(id), m_entityCount(scene_desc.entityCount),
	m_mainCameraId(scene_desc.mainCameraId),
	m_lightStartId(scene_desc.startLightId),
	m_lightCount(scene_desc.lightCount)
{
	m_pEntityCollection = new Entity[m_entityCount];
	memcpy(m_pEntityCollection, scene_desc.entityCollection, sizeof(Entity) * m_entityCount);
	SAFE_DELETE_ARRAY(scene_desc.entityCollection);
}
