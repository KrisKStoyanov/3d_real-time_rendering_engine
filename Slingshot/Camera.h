#pragma once
#include "Transform.h"

struct CAMERA_DESC {
	float verticalFovAngle = 75.0f; //in degrees, converted to radians during view setup
	float lenseWidth = 1280;
	float lenseHeight = 720;
	float nearClipDist = 1.0f;
	float farClipDist = 1000.0f;
	float translationSpeed = 32.0f;
	float rotationSpeed = 0.1f;
};

class Camera
{
public:
	static Camera* Create(
		CAMERA_DESC& camera_desc, 
		Transform& transform);

	DirectX::XMMATRIX GetViewMatrix();
	DirectX::XMMATRIX GetProjectionMatrix();

	void GetMouseCoord(int& mouseX, int& mouseY);
	void SetMouseCoord(int mouseX, int mouseY);

	float GetRotationSpeed();

	bool GetRotateStatus();
	void SetRotateStatus(bool rotate);

	float GetTranslationSpeed();

	void Update(Transform& transform);
private:
	Camera(
		CAMERA_DESC& camera_desc,
		Transform& transform);

	DirectX::XMMATRIX m_viewMatrix;
	DirectX::XMMATRIX m_projectionMatrix;

	int m_lastMouseX;
	int m_lastMouseY;

	bool m_rotate;
	float m_rotationSpeed;
	float m_translationSpeed;
};

