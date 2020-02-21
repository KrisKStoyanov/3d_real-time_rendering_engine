#pragma once
#include "Transform.h"

struct CAMERA_DESC {
	float verticalFovAngle = 75.0f; //in degrees, converted to radians during setup
	float lenseWidth = 1280;
	float lenseHeight = 720;
	float nearClipDist = 1.0f;
	float farClipDist = 1000.0f;
};

class Camera
{
public:
	static Camera* Create(CAMERA_DESC& camera_desc, Transform& transform);

	DirectX::XMMATRIX GetViewMatrix();
	DirectX::XMMATRIX GetProjectionMatrix();

	void OnFrameRender(Transform& transform);
private:
	Camera(CAMERA_DESC& camera_desc, Transform& transform);

	DirectX::XMMATRIX m_viewMatrix;
	DirectX::XMMATRIX m_projectionMatrix;
};

