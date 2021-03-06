#pragma once

#include <UGM/UGM.h>

struct CanvasData {
	std::vector<Ubpa::pointf2> points;
	Ubpa::valf2 scrolling{ 0.f,0.f };
	bool opt_enable_grid{ true };
	bool opt_enable_context_menu{ true };
	bool generate_line{ false };

	bool opt_proc[4]{ true,true,true,true };

	float lambda=1.0f;
	float b0 = 0.0f;
	int m = 2; //函数的最高次数
	float sigma = 1.0f;

	float cur_lambda = 1.0f;
	float cur_b0 = 0.0f;
	int cur_m = 2; //函数的最高次数
	float cur_sigma = 1.0f;

	std::vector<float> panel_x;
	std::vector<float> panel_y[4];
	
	
};

#include "details/CanvasData_AutoRefl.inl"
