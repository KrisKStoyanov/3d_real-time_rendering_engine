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
	GEOMETRY_DESC SetupScene(Scene& scene, const int entityCount);
	void Shutdown();

	void CreatePlane(Entity& entity, PLANE_DESC& plane_desc, MATERIAL_DESC& material_desc);
	void CreateCube(Entity& entity, CUBE_DESC& cube_desc, MATERIAL_DESC& material_desc);
	void CreateSphere(Entity& entity, SPHERE_DESC& sphere_desc, MATERIAL_DESC& material_desc);

	LRESULT CALLBACK HandleWindowMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
	inline Renderer* GetRenderer() { return m_pRenderer; }
private:
	Engine() :
		m_pWindow(nullptr),
		m_pRenderer(nullptr),
		m_pTimer(nullptr),
		m_pScene(nullptr),
		m_isRunning(false)
	{}
	Window* m_pWindow;	
	Renderer* m_pRenderer;
	Timer* m_pTimer;
	Scene* m_pScene;

	bool m_isRunning;

};