#pragma once
#include "Transform.h"

struct CAMERA_DESC {
	float fov;
	float lenseWidth;
	float lenseHeight;
	float nearClipDist;
	float farClipDist;
	TRANSFORM_DESC* transform_desc;
	CAMERA_DESC(
		float _fov,
		float _lenseWidth,
		float _lenseHeight,
		float _nearClipDist,
		float _farClipDist,
		TRANSFORM_DESC* _transform_desc = nullptr) :
		fov(_fov),
		lenseWidth(_lenseWidth),
		lenseHeight(_lenseHeight),
		nearClipDist(_nearClipDist),
		farClipDist(_farClipDist),
		transform_desc(_transform_desc) 
	{}
};

class Camera
{
public:
	static Camera* Create(CAMERA_DESC* camera_desc);

	DirectX::XMMATRIX GetViewMatrix();
	DirectX::XMMATRIX GetProjectionMatrix();

	Transform* GetTransform();
private:
	Camera(CAMERA_DESC* camera_desc);

	DirectX::XMMATRIX m_viewMatrix;
	DirectX::XMMATRIX m_projectionMatrix;

	Transform* m_pTransform;
};

