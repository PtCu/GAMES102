#pragma once

#include <UGM/UGM.h>

enum class Narrow_Type {
	chaikin,cubic,quad
};

enum class Para_Type {
	Uniform, Chord, Centripetal, Foley
};

struct CanvasData {
	std::vector<Ubpa::pointf2> points;
	std::vector<Ubpa::pointf2> output_points;
	bool generate_line{ false };
	bool opt_enable_grid{ true };
	bool opt_enable_context_menu{ true };
	Ubpa::valf2 scrolling{ 0.f,0.f };
	bool isOpen{ false };
	int times{ 10 };
	int cur_times{ 10 };
	Narrow_Type narrow_type{ Narrow_Type::chaikin};
	Narrow_Type cur_narrow_type{ Narrow_Type::chaikin };
	std::function<void(std::vector<float>&, std::vector<Ubpa::pointf2>&)> param_f;
};
#include "details/CanvasData_AutoRefl.inl"
