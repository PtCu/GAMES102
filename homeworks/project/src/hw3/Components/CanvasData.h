#pragma once

#include <UGM/UGM.h>

struct CanvasData {
	std::vector<Ubpa::pointf2> input_points;
	std::vector<Ubpa::pointf2> output_points;

	Ubpa::valf2 scrolling{ 0.f,0.f };
	bool opt_enable_grid{ true };
	bool opt_enable_context_menu{ true };
	bool generate_line{ false };

	bool opt_proc[4]{ true,true,true,true };
	bool opt_param[4]{ true,false,false,false };

	float lambda = 1.0f;
	float b0 = 0.0f;
	int m = 100; //函数的最高次数
	float sigma = 1.0f;

	float cur_lambda = 1.0f;
	float cur_b0 = 0.0f;
	int cur_m = 100; //函数的最高次数
	float cur_sigma = 1.0f;

	std::function<void(std::vector<float>&, std::vector<float>&, std::vector<float>& , int num,float lambda, int m, float b0, float sigma)> fitting_f;
	std::function<void(std::vector<float>&, std::vector<Ubpa::pointf2>&)> param_f;



};

#include "details/CanvasData_AutoRefl.inl"