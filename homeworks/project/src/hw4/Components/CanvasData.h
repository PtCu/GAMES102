#pragma once

#include <UGM/UGM.h>

enum class State {
	init, adding_point, editing, done
};

enum class Para_Type {
	Uniform, Chord, Centripetal, Foley
};

enum class Edit_State {
	init, dragging_point,dragging_tan
};

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


struct CanvasData {
	std::vector<Ubpa::pointf2> points;
	std::vector<Ubpa::pointf2> ltangent;
	std::vector<Ubpa::pointf2> rtangent;
	std::vector<Ratio> tangent_ratio;
	std::vector<Slope> xk;
	std::vector<Slope> yk;
	Ubpa::valf2 scrolling{ 0.f,0.f };

	Para_Type para_type{ Para_Type::Uniform };

	bool opt_enable_grid{ true };
	bool opt_enable_context_menu{ true };
	bool adding_point{ false };

	bool stop_dragg{ true };
	int fitting_type{ 0 };
	bool enable_add_point{ true };
	int edit_point{ 0 };
	int editing_index = -1;
	bool enable_move_point{ false };
	int editing_tan_index = 0; // < 0 is left, > 0 is right, real index = this - 1
	bool enable_move_tan{ false };

	bool c_type{ false }; //0 is c0, 1 is c1
	Edit_State cur_edit_state{ Edit_State::init };
	Edit_State last_edit_state{ Edit_State::init };
	State cur_state{ State::adding_point };
	State last_state{State::init};
	void pop_back() {
		points.pop_back();
		ltangent.pop_back();
		rtangent.pop_back();
		
	}
	void clear() {
		points.clear();
		ltangent.clear();
		rtangent.clear();
		
	}
	void push_back(const Ubpa::pointf2& p) {
		points.push_back(p);
		ltangent.push_back(Ubpa::pointf2());
		rtangent.push_back(Ubpa::pointf2());
		
	}
	//Ubpa::pointf2 edited_point_co;
	std::function<void(std::vector<float>&, std::vector<Ubpa::pointf2>&)> param_f;
};
#include "details/CanvasData_AutoRefl.inl"
