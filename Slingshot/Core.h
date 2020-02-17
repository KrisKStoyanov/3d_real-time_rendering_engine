#pragma once
#include "Renderer.h"

class Core
{
public:
	static Core* Create(HWND hWnd);
	LRESULT CALLBACK HandleMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
	bool InitializeRenderer(HWND hWnd, RENDERER_DESC* core_desc);
	void LoadStage(Stage* stage);
	bool OnUpdate(void);
	void Shutdown(void);

	Renderer* GetRenderer();
	Stage* GetStage();
private:
	Core(HWND hWnd);
	bool m_isActive;
	HWND m_hWnd;

	Renderer* m_pRenderer;

	Stage* m_pStage;
	Entity* m_pStageEntities;
	unsigned int m_stageEntityCount;
};

