#pragma once
#include "Transform.h"

struct CAMERA_DESC {
	float verticalFovAngle; //angle in degrees, converted to radians during setup
	float lenseWidth;
	float lenseHeight;
	float nearClipDist;
	float farClipDist;
	CAMERA_DESC(
		float _verticalFovAngle,
		float _lenseWidth,
		float _lenseHeight,
		float _nearClipDist,
		float _farClipDist) :
		verticalFovAngle(_verticalFovAngle),
		lenseWidth(_lenseWidth),
		lenseHeight(_lenseHeight),
		nearClipDist(_nearClipDist),
		farClipDist(_farClipDist)
	{}
};

class Camera
{
public:
	static Camera* Create(CAMERA_DESC* camera_desc, Transform* transform);

	DirectX::XMMATRIX GetViewMatrix();
	DirectX::XMMATRIX GetProjectionMatrix();

	void OnFrameRender(Transform* transform);
private:
	Camera(CAMERA_DESC* camera_desc, Transform* transform);

	DirectX::XMMATRIX m_viewMatrix;
	DirectX::XMMATRIX m_projectionMatrix;
};

