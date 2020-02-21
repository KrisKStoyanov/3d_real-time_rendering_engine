#pragma once
#include "Renderer.h"

class Core
{
public:
	static Core* Create(HWND hWnd);
	LRESULT CALLBACK HandleMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
	void LoadStage(Stage* stage);
	bool OnUpdate(Renderer& renderer);
	void Shutdown(void);

	Stage* GetStage();
private:
	Core(HWND hWnd);
	bool m_isActive;
	HWND m_hWnd;

	Stage* m_pStage;
	Entity* m_pStageEntities;
	unsigned int m_stageEntityCount;
};

