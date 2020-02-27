#pragma once
#include "Renderer.h"
#include "Timer.h"
#include <iostream>

class Engine {
public:
	static Engine& Get() {
		static Engine instance;
		return instance;
	}

	bool Initialize(
		WINDOW_DESC& window_desc,
		RENDERER_DESC& renderer_desc);
	int Run();
	void EditStage(Stage& stage);
	void Shutdown();

	void CreatePlane(Entity& entity, float width, float length, DirectX::XMFLOAT4 color);
	void CreateCube(Entity& entity, float width, float height, float length, DirectX::XMFLOAT4 color);
	void CreateSphere(Entity& entity, unsigned int slices, unsigned int stacks, float radius, DirectX::XMFLOAT4 color);

	LRESULT CALLBACK HandleWindowMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
	Renderer* GetRenderer();
private:
	Engine() :
		m_pWindow(nullptr),
		m_pRenderer(nullptr),
		m_pTimer(nullptr),
		m_pStage(nullptr),
		m_isRunning(false)
	{}
	Window* m_pWindow;	
	Renderer* m_pRenderer;
	Timer* m_pTimer;
	Stage* m_pStage;

	bool m_isRunning;

};