#pragma once
#include <DirectXMath.h>

struct TRANSFORM_DESC {
	DirectX::XMFLOAT4 position;
	DirectX::XMFLOAT4 rotation;
	DirectX::XMFLOAT4 scale;
	DirectX::XMFLOAT4 forwardDir;
	DirectX::XMFLOAT4 rightDir;
	DirectX::XMFLOAT4 upDir;
	TRANSFORM_DESC(
		DirectX::XMFLOAT4 _position = DirectX::XMFLOAT4(0.0f, 0.0f, 0.0f, 1.0f),
		DirectX::XMFLOAT4 _rotation = DirectX::XMFLOAT4(0.0f, 0.0f, 0.0f, 1.0f),
		DirectX::XMFLOAT4 _scale = DirectX::XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f),
		DirectX::XMFLOAT4 _forwardDir = DirectX::XMFLOAT4(0.0f, 0.0f, 1.0f, 0.0f), //must differ from position for camera view matrix setup
		DirectX::XMFLOAT4 _rightDir = DirectX::XMFLOAT4(1.0f, 0.0f, 0.0f, 0.0f),
		DirectX::XMFLOAT4 _upDir = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 0.0f)) :
		position(_position),
		rotation(_rotation),
		scale(_scale),
		forwardDir(_forwardDir),
		rightDir(_rightDir),
		upDir(_upDir)
	{}
};

class Transform
{
public:
	static Transform* Create(TRANSFORM_DESC* transform_desc);

	DirectX::XMMATRIX GetWorldMatrix();

	DirectX::XMVECTOR GetPosition();
	DirectX::XMVECTOR GetRotation();
	DirectX::XMVECTOR GetScale();

	DirectX::XMVECTOR GetForwardDir();
	DirectX::XMVECTOR GetRightDir();
	DirectX::XMVECTOR GetUpDir();

	void OnFrameRender();

	void Translate(DirectX::XMVECTOR translation);
	void Rotate(DirectX::XMVECTOR rotation);
	void Scale(DirectX::XMVECTOR scale);
private:
	Transform(TRANSFORM_DESC* transform_desc);

	DirectX::XMMATRIX m_worldMatrix;

	DirectX::XMMATRIX m_translatioMatrix;
	DirectX::XMMATRIX m_rotationMatrix;
	DirectX::XMMATRIX m_scalingMatrix;

	DirectX::XMVECTOR m_position;
	DirectX::XMVECTOR m_rotation;
	DirectX::XMVECTOR m_scale;

	DirectX::XMVECTOR m_forwardDir;
	DirectX::XMVECTOR m_rightDir;
	DirectX::XMVECTOR m_upDir;

	DirectX::XMVECTOR m_defaultForwardDir;
	DirectX::XMVECTOR m_defaultRightDir;
	DirectX::XMVECTOR m_defaultUpDir;
};

