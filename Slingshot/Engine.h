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

	void CreatePlane(Entity& entity, float width, float length, MATERIAL_DESC& material_desc);
	void CreateCube(Entity& entity, float width, float height, float length, MATERIAL_DESC& material_desc);
	void CreateSphere(Entity& entity, unsigned int slices, unsigned int stacks, float radius, MATERIAL_DESC& material_desc);

	LRESULT CALLBACK HandleWindowMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
	inline Renderer* GetRenderer() { return m_pRenderer; }
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