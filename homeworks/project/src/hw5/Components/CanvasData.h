#pragma once

#include <UGM/UGM.h>

struct Slope
{
	float l;
	float r;
};

struct Ratio
{
	float l;
	float r;
	Ratio() { l = r = 1.f; }
};

struct CanvasData
{
	std::vector<Ubpa::pointf2> points;
	std::vector<Ubpa::pointf2> ltangent;
	std::vector<Ubpa::pointf2> rtangent;
	std::vector<Ratio> tangent_ratio;
	std::vector<Slope> xk;
	std::vector<Slope> yk;
	Ubpa::valf2 scrolling{ 0.f, 0.f };
	bool opt_enable_grid{ true };
	bool opt_enable_context_menu{ true };
	bool adding_line{ false };
	int param_type{ 0 };
	int fitting_type{ 0 };
	bool enable_add_point{ true };
	bool adding_last_point{ false };
	int edit_point{ 0 };
	int editing_index = -1;
	bool enable_move_point{ false };
	int editing_tan_index = 0; // < 0 is left, > 0 is right, real index = this - 1
	bool enable_move_tan{ false };
	void pop_back()
	{
		points.pop_back();
		ltangent.pop_back();
		rtangent.pop_back();
		xk.pop_back();
		yk.pop_back();
		tangent_ratio.pop_back();
	}
	void clear()
	{
		points.clear();
		ltangent.clear();
		rtangent.clear();
		xk.clear();
		yk.clear();
		tangent_ratio.clear();
	}
	void push_back(const Ubpa::pointf2& p)
	{
		points.push_back(p);
		ltangent.push_back(Ubpa::pointf2());
		rtangent.push_back(Ubpa::pointf2());
		xk.push_back(Slope());
		yk.push_back(Slope());
		tangent_ratio.push_back(Ratio());
	}
	//Ubpa::pointf2 edited_point_co;
};
#include "details/CanvasData_AutoRefl.inl"
