#include "CanvasSystem.h"

#include "../Components/CanvasData.h"

#include <_deps/imgui/imgui.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "spdlog/spdlog.h"

using namespace Ubpa;

constexpr int sample_num = 500;
constexpr float base_tangent_len = 50.f;
constexpr float t_step = 1.f / (sample_num - 1);
constexpr float point_radius = 3.f;
constexpr ImU32 line_col = IM_COL32(39, 117, 182, 255);
constexpr ImU32 edit_line_col = IM_COL32(39, 117, 182, 100);
constexpr ImU32 normal_point_col = IM_COL32(255, 0, 0, 255);
constexpr ImU32 select_point_col = IM_COL32(255, 0, 0, 100);
constexpr ImU32 slope_col = IM_COL32(122, 115, 116, 255);
constexpr ImU32 select_slope_col = IM_COL32(122, 115, 116, 100);

// four methods to parameterize x, y
void UniformParameterize(std::vector<Ubpa::pointf2>& xt, std::vector<Ubpa::pointf2>& yt);
void ChordalParameterize(std::vector<Ubpa::pointf2>& xt, std::vector<Ubpa::pointf2>& yt);
void CentripetalParameterize(std::vector<Ubpa::pointf2>& xt, std::vector<Ubpa::pointf2>& yt);
void FoleyParameterize(std::vector<Ubpa::pointf2>& xt, std::vector<Ubpa::pointf2>& yt);
// function points for parameterization
void (*ParamFunc[4])(std::vector<Ubpa::pointf2>&, std::vector<Ubpa::pointf2>&) = {
	UniformParameterize, ChordalParameterize, CentripetalParameterize, FoleyParameterize };

inline void AddLine(const ImVec2& p1, const ImVec2& p2, ImDrawList* draw_list, bool edit_line = false)
{
	draw_list->AddLine(p1, p2, (edit_line ? edit_line_col : line_col), 2.f);
}

//void draw(std::vector<Ubpa::pointf2>&, CanvasData*, ImDrawList*, const ImVec2&, bool edit_flag = false, bool with_slope = false);
void draw(std::vector<Ubpa::pointf2>& points,
	std::vector<Ubpa::pointf2>& ltangent,
	std::vector<Ubpa::pointf2>& rtangent,
	CanvasData*, ImDrawList*, const ImVec2&,
	int edit_flag = 0, bool with_slope = false);
void drawWithBezier(std::vector<Ubpa::pointf2>& points,
	std::vector<Ubpa::pointf2>& ltangent,
	std::vector<Ubpa::pointf2>& rtangent,
	CanvasData*, ImDrawList*, const ImVec2&,
	int edit_flag = 0);

void CubicSpline(std::vector<Ubpa::pointf2>&, std::vector<Ubpa::pointf2>&, std::vector<Slope>&, bool);
void SlopeSpline(std::vector<Ubpa::pointf2>&, std::vector<Ubpa::pointf2>&, std::vector<Slope>&);

void CanvasSystem::OnUpdate(Ubpa::UECS::Schedule& schedule)
{
	schedule.RegisterCommand([](Ubpa::UECS::World* w) {
		auto data = w->entityMngr.GetSingleton<CanvasData>();
		if (!data)
			return;

		if (ImGui::Begin("Canvas"))
		{
			ImGui::Checkbox("Enable grid", &data->opt_enable_grid);
			ImGui::Checkbox("Enable context menu", &data->opt_enable_context_menu);
			ImGui::Text("Mouse Left: drag to add points,\nMouse Right: drag to scroll, click for context menu, \nMouse Middle: fixed point.");

			if (ImGui::RadioButton("Cubic Spline ", data->fitting_type == 0))
				data->fitting_type = 0;
			ImGui::SameLine();
			if (ImGui::RadioButton("Cubic Bezier ", data->fitting_type == 1))
				data->fitting_type = 1;

			// Using InvisibleButton() as a convenience 1) it will advance the layout cursor and 2) allows us to use IsItemHovered()/IsItemActive()
			ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();	   // ImDrawList API uses screen coordinates!
			ImVec2 canvas_sz = ImGui::GetContentRegionAvail(); // Resize canvas to what's available
			if (canvas_sz.x < 50.0f)
				canvas_sz.x = 50.0f;
			if (canvas_sz.y < 50.0f)
				canvas_sz.y = 50.0f;
			ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);
			bool change_flag = false;

			// Draw border and background color
			ImGuiIO& io = ImGui::GetIO();
			ImDrawList* draw_list = ImGui::GetWindowDrawList();
			draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));
			draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

			// This will catch our interactions
			ImGui::InvisibleButton("canvas", canvas_sz, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
			const bool is_hovered = ImGui::IsItemHovered();											 // Hovered
			const bool is_active = ImGui::IsItemActive();											 // Held
			const ImVec2 origin(canvas_p0.x + data->scrolling[0], canvas_p0.y + data->scrolling[1]); // Lock scrolled origin
			const pointf2 mouse_pos_in_canvas(io.MousePos.x - origin.x, io.MousePos.y - origin.y);

			// Add point
			if (is_hovered)
			{
				if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle) && data->edit_point == 0)
				{
					if (data->points.size() > 0)
						data->pop_back();
					// spdlog::info("d:{}", data->points.size());
					data->enable_add_point = false;
					data->adding_last_point = false;
				}
				if (data->enable_add_point)
				{
					change_flag = true;
					if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))
					{
						data->push_back(mouse_pos_in_canvas);
					}
					else
					{
						if (data->points.size() > 0 && data->adding_last_point)
							data->pop_back();
						data->adding_last_point = true;
						data->push_back(mouse_pos_in_canvas);
					}
				}
			}

			if (data->edit_point != 0)
			{
				if (!data->enable_move_point)
				{
					data->editing_index = -1;
					for (int i = 0; i < data->points.size(); i++)
					{
						if (std::abs(data->points[i][0] - mouse_pos_in_canvas[0]) < point_radius && std::abs(data->points[i][1] - mouse_pos_in_canvas[1]) < point_radius)
						{
							data->editing_index = i;
							break;
						}
					}
				}

				if (data->editing_index != -1)
				{
					if (data->enable_move_point)
					{
						draw_list->AddCircleFilled(ImVec2(origin.x + mouse_pos_in_canvas[0], origin.y + mouse_pos_in_canvas[1]), point_radius, select_point_col);
						std::vector<Ubpa::pointf2> temp_p = data->points;
						temp_p[data->editing_index] = mouse_pos_in_canvas;
						if (data->fitting_type == 0)
						{
							draw(temp_p, data->ltangent, data->rtangent, data, draw_list, origin, 1);
						}
						else if (data->fitting_type == 1)
						{
							std::vector<Ubpa::pointf2> temp_lt = data->ltangent;
							std::vector<Ubpa::pointf2> temp_rt = data->rtangent;
							drawWithBezier(temp_p, temp_lt, temp_rt, data, draw_list, origin, 1);
						}
						if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))
						{
							data->points[data->editing_index] = mouse_pos_in_canvas;
							data->editing_index = -1;
							data->enable_move_point = false;
							change_flag = true;
						}
					}
					else
					{
						draw_list->AddCircleFilled(ImVec2(origin.x + data->points[data->editing_index][0], origin.y + data->points[data->editing_index][1]), point_radius + 2.f, select_point_col);

						if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))
						{
							data->enable_move_point = true;
							// spdlog::info("c3:{}", data->points.size());
						}
						else
							data->editing_index = -1;
					}
				}
				// G0
				if (data->edit_point == 2)
				{
					if (!data->enable_move_tan)
					{
						data->editing_tan_index = 0;
						for (int i = 0; i < data->points.size(); i++)
						{
							if (std::abs(data->ltangent[i][0] - mouse_pos_in_canvas[0]) < point_radius && std::abs(data->ltangent[i][1] - mouse_pos_in_canvas[1]) < point_radius)
							{
								data->editing_tan_index = -(i + 1);
								break;
							}
							if (std::abs(data->rtangent[i][0] - mouse_pos_in_canvas[0]) < point_radius && std::abs(data->rtangent[i][1] - mouse_pos_in_canvas[1]) < point_radius)
							{
								data->editing_tan_index = i + 1;
								break;
							}
						}
					}
					if (data->editing_tan_index < 0)
					{
						//spdlog::info("{} {}", -1 - data->editing_tan_index, data->ltangent.size());
						if (data->enable_move_tan)
						{
							draw_list->AddCircleFilled(ImVec2(origin.x + mouse_pos_in_canvas[0], origin.y + mouse_pos_in_canvas[1]), point_radius, select_point_col);
							std::vector<Ubpa::pointf2> temp_lt = data->ltangent;
							temp_lt[-1 - data->editing_tan_index] = mouse_pos_in_canvas;
							draw(data->points, temp_lt, data->rtangent, data, draw_list, origin, 2, true);

							if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))
							{
								data->ltangent[-1 - data->editing_tan_index] = mouse_pos_in_canvas;
								data->editing_tan_index = 0;
								data->enable_move_tan = false;
								// change_flag = true;
							}
						}
						else
						{
							draw_list->AddCircleFilled(ImVec2(origin.x + data->ltangent[-1 - data->editing_tan_index][0], origin.y + data->ltangent[-1 - data->editing_tan_index][1]), point_radius + 2.f, select_point_col);

							if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))
							{
								data->enable_move_tan = true;
							}
							else
								data->editing_tan_index = 0;
						}
					}

					if (data->editing_tan_index > 0)
					{
						//spdlog::info("{} {}", -1 + data->editing_tan_index, data->rtangent.size());
						if (data->enable_move_tan)
						{
							draw_list->AddCircleFilled(ImVec2(origin.x + mouse_pos_in_canvas[0], origin.y + mouse_pos_in_canvas[1]), point_radius, select_point_col);
							std::vector<Ubpa::pointf2> temp_rt = data->rtangent;
							temp_rt[data->editing_tan_index - 1] = mouse_pos_in_canvas;
							draw(data->points, data->ltangent, temp_rt, data, draw_list, origin, 2, true);

							if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))
							{
								data->rtangent[data->editing_tan_index - 1] = mouse_pos_in_canvas;
								data->editing_tan_index = 0;
								data->enable_move_tan = false;
								// change_flag = true;
							}
						}
						else
						{
							draw_list->AddCircleFilled(ImVec2(origin.x + data->rtangent[data->editing_tan_index - 1][0], origin.y + data->rtangent[data->editing_tan_index - 1][1]), point_radius + 2.f, select_point_col);

							if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))
							{
								data->enable_move_tan = true;
							}
							else
								data->editing_tan_index = 0;
						}
					}
				}
				// G1
				if (data->edit_point == 3)
				{
					if (!data->enable_move_tan)
					{
						data->editing_tan_index = 0;
						for (int i = 0; i < data->points.size(); i++)
						{
							if (std::abs(data->ltangent[i][0] - mouse_pos_in_canvas[0]) < point_radius && std::abs(data->ltangent[i][1] - mouse_pos_in_canvas[1]) < point_radius)
							{
								data->editing_tan_index = -(i + 1);
								break;
							}
							if (std::abs(data->rtangent[i][0] - mouse_pos_in_canvas[0]) < point_radius && std::abs(data->rtangent[i][1] - mouse_pos_in_canvas[1]) < point_radius)
							{
								data->editing_tan_index = i + 1;
								break;
							}
						}
					}
					if (data->editing_tan_index < 0)
					{
						//spdlog::info("{} {}", -1 - data->editing_tan_index, data->ltangent.size());
						if (data->enable_move_tan)
						{
							draw_list->AddCircleFilled(ImVec2(origin.x + mouse_pos_in_canvas[0], origin.y + mouse_pos_in_canvas[1]), point_radius, select_point_col);
							std::vector<Ubpa::pointf2> temp_lt = data->ltangent;
							std::vector<Ubpa::pointf2> temp_rt = data->rtangent;
							temp_lt[-1 - data->editing_tan_index] = mouse_pos_in_canvas;
							float ratio_distance = data->points[-1 - data->editing_tan_index].distance(temp_rt[-1 - data->editing_tan_index]) /
								data->points[-1 - data->editing_tan_index].distance(mouse_pos_in_canvas);
							float x = mouse_pos_in_canvas[0] - data->points[-1 - data->editing_tan_index][0];
							x = data->points[-1 - data->editing_tan_index][0] - x * ratio_distance;
							float y = mouse_pos_in_canvas[1] - data->points[-1 - data->editing_tan_index][1];
							y = data->points[-1 - data->editing_tan_index][1] - y * ratio_distance;
							temp_rt[-1 - data->editing_tan_index] = Ubpa::pointf2(x, y);
							draw(data->points, temp_lt, temp_rt, data, draw_list, origin, 2, true);

							if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))
							{
								data->ltangent[-1 - data->editing_tan_index] = mouse_pos_in_canvas;
								data->rtangent[-1 - data->editing_tan_index] = Ubpa::pointf2(x, y);
								data->editing_tan_index = 0;
								data->enable_move_tan = false;
								// change_flag = true;
							}
						}
						else
						{
							draw_list->AddCircleFilled(ImVec2(origin.x + data->ltangent[-1 - data->editing_tan_index][0], origin.y + data->ltangent[-1 - data->editing_tan_index][1]), point_radius + 2.f, select_point_col);

							if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))
							{
								data->enable_move_tan = true;
							}
							else
								data->editing_tan_index = 0;
						}
					}

					if (data->editing_tan_index > 0)
					{
						//spdlog::info("{} {}", -1 + data->editing_tan_index, data->rtangent.size());
						if (data->enable_move_tan)
						{
							draw_list->AddCircleFilled(ImVec2(origin.x + mouse_pos_in_canvas[0], origin.y + mouse_pos_in_canvas[1]), point_radius, select_point_col);
							std::vector<Ubpa::pointf2> temp_lt = data->ltangent;
							std::vector<Ubpa::pointf2> temp_rt = data->rtangent;
							temp_rt[data->editing_tan_index - 1] = mouse_pos_in_canvas;
							float ratio_distance = data->points[data->editing_tan_index - 1].distance(temp_lt[data->editing_tan_index - 1]) /
								data->points[data->editing_tan_index - 1].distance(mouse_pos_in_canvas);
							float x = mouse_pos_in_canvas[0] - data->points[data->editing_tan_index - 1][0];
							x = data->points[data->editing_tan_index - 1][0] - x * ratio_distance;
							float y = mouse_pos_in_canvas[1] - data->points[data->editing_tan_index - 1][1];
							y = data->points[data->editing_tan_index - 1][1] - y * ratio_distance;
							temp_lt[data->editing_tan_index - 1] = Ubpa::pointf2(x, y);
							draw(data->points, temp_lt, temp_rt, data, draw_list, origin, 2, true);

							if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))
							{
								data->ltangent[data->editing_tan_index - 1] = Ubpa::pointf2(x, y);
								data->rtangent[data->editing_tan_index - 1] = mouse_pos_in_canvas;
								data->editing_tan_index = 0;
								data->enable_move_tan = false;
								// change_flag = true;
							}
						}
						else
						{
							draw_list->AddCircleFilled(ImVec2(origin.x + data->rtangent[data->editing_tan_index - 1][0], origin.y + data->rtangent[data->editing_tan_index - 1][1]), point_radius + 2.f, select_point_col);

							if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))
							{
								data->enable_move_tan = true;
							}
							else
								data->editing_tan_index = 0;
						}
					}
				}
			}

			const float mouse_threshold_for_pan = data->opt_enable_context_menu ? -1.0f : 0.0f;
			if (is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Right, mouse_threshold_for_pan))
			{
				data->scrolling[0] += io.MouseDelta.x;
				data->scrolling[1] += io.MouseDelta.y;
			}

			// Context menu (under default mouse threshold)
			ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
			if (data->opt_enable_context_menu && ImGui::IsMouseReleased(ImGuiMouseButton_Right) && drag_delta.x == 0.0f && drag_delta.y == 0.0f)
				ImGui::OpenPopupContextItem("context");
			if (ImGui::BeginPopup("context"))
			{
				if (ImGui::MenuItem("Remove one", NULL, false, data->points.size() > 0))
				{
					data->pop_back();
					// spdlog::info("{}", data->points.size());
					if (data->points.size() < 2)
					{
						data->clear();
						data->enable_add_point = true;
						data->edit_point = 0;
					}
				}
				if (ImGui::MenuItem("Remove all", NULL, false, data->points.size() > 0))
				{
					data->clear();
					data->enable_add_point = true;
					data->edit_point = 0;
				}
				if (!data->enable_add_point)
				{
					if (data->edit_point == 0)
					{
						if (ImGui::MenuItem("Edit Points' position", NULL, false, data->points.size() > 0))
							data->edit_point = 1;
						if (data->fitting_type == 0)
						{
							if (ImGui::MenuItem("Edit Points' slope (G0)", NULL, false))
								data->edit_point = 2;
							if (ImGui::MenuItem("Edit Points' slope (G1)", NULL, false))
								data->edit_point = 3;
						}
					}
					else
					{
						if (ImGui::MenuItem("Cancel Edit Points", NULL, false, data->points.size() > 0))
							data->edit_point = 0;
					}
					if (ImGui::MenuItem("Enable Add Points", NULL, false))
						data->enable_add_point = true, data->edit_point = 0;
				}
				ImGui::EndPopup();
			}

			// Draw grid + all lines in the canvas
			draw_list->PushClipRect(canvas_p0, canvas_p1, true);
			if (data->opt_enable_grid)
			{
				const float GRID_STEP = 64.0f;
				for (float x = fmodf(data->scrolling[0], GRID_STEP); x < canvas_sz.x; x += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x + x, canvas_p0.y), ImVec2(canvas_p0.x + x, canvas_p1.y), IM_COL32(200, 200, 200, 200));
				for (float y = fmodf(data->scrolling[1], GRID_STEP); y < canvas_sz.y; y += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x, canvas_p0.y + y), ImVec2(canvas_p1.x, canvas_p0.y + y), IM_COL32(200, 200, 200, 200));
			}
			// Draw points
			for (int n = 0; n < data->points.size(); n++)
				draw_list->AddCircleFilled(ImVec2(origin.x + data->points[n][0], origin.y + data->points[n][1]), point_radius, normal_point_col);

			if (data->points.size() < 2)
				data->enable_add_point = true, data->edit_point = 0;
			else
			{
				// void (*ParamFunc)(std::vector<Ubpa::pointf2>&, std::vector<Ubpa::pointf2>&) = UniformParameterize;
				if (data->fitting_type == 0)
				{
					draw(data->points, data->ltangent, data->rtangent, data, draw_list, origin, false, !change_flag);
				}
				else if (data->fitting_type == 1)
				{
					drawWithBezier(data->points, data->ltangent, data->rtangent, data, draw_list, origin, false);
				}
			}

			draw_list->PopClipRect();
		}

		ImGui::End();
		});
}

constexpr int Bezier_num = 100;
void DeCasteljau(std::vector<Ubpa::pointf2>& ret, Ubpa::pointf2* p)
{
	const float step = 1.f / (Bezier_num - 1);
	for (float t = 0.f; t <= 1.f; t += step)
	{
		float x = std::pow((1 - t), 3) * p[0][0] + 3 * t * (1 - t) * (1 - t) * p[1][0] + 3 * t * t * (1 - t) * p[2][0] + t * t * t * p[3][0];
		float y = std::pow((1 - t), 3) * p[0][1] + 3 * t * (1 - t) * (1 - t) * p[1][1] + 3 * t * t * (1 - t) * p[2][1] + t * t * t * p[3][1];
		ret.push_back(Ubpa::pointf2(x, y));
	}
}

void drawWithBezier(std::vector<Ubpa::pointf2>& points,
	std::vector<Ubpa::pointf2>& ltangent,
	std::vector<Ubpa::pointf2>& rtangent,
	CanvasData* data, ImDrawList* draw_list, const ImVec2& origin,
	int edit_flag)
{
	size_t n = points.size();
	if (n == 2)
	{
		AddLine(ImVec2(origin.x + points[0][0], origin.y + points[0][1]),
			ImVec2(origin.x + points[1][0], origin.y + points[1][1]), draw_list, edit_flag);
		return;
	}
	for (int i = 1; i < n - 1; i++)
	{
		float dx = points[i + 1][0] - points[i - 1][0];
		float dy = points[i + 1][1] - points[i - 1][1];
		rtangent[i] = Ubpa::pointf2(points[i][0] + dx / 6.f, points[i][1] + dy / 6.f);
		ltangent[i] = Ubpa::pointf2(points[i][0] - dx / 6.f, points[i][1] - dy / 6.f);
	}
	rtangent[0] = Ubpa::pointf2(points[0][0] + (points[1][0] - points[2][0]) / 6.f, points[0][1] + (points[1][1] - points[2][1]) / 6.f);
	ltangent[n - 1] = Ubpa::pointf2(points[n - 1][0] - (points[n - 2][0] - points[n - 3][0]) / 6.f, points[n - 1][1] - (points[n - 2][1] - points[n - 3][1]) / 6.f);
	for (int i = 0; i < n - 1; i++)
	{
		Ubpa::pointf2 control_points[4] = { points[i], rtangent[i], ltangent[i + 1], points[i + 1] };
		std::vector<Ubpa::pointf2> p;
		DeCasteljau(p, control_points);
		for (int j = 0; j < p.size() - 1; j++)
			AddLine(ImVec2(origin.x + p[j][0], origin.y + p[j][1]),
				ImVec2(origin.x + p[j + 1][0], origin.y + p[j + 1][1]), draw_list, edit_flag);
	}
}

void draw(std::vector<Ubpa::pointf2>& points,
	std::vector<Ubpa::pointf2>& ltangent,
	std::vector<Ubpa::pointf2>& rtangent,
	CanvasData* data,
	ImDrawList* draw_list,
	const ImVec2& origin,
	int edit_flag,
	bool with_slope)
{
	// Parameterize
	std::vector<Ubpa::pointf2> xt = points;
	std::vector<Ubpa::pointf2> yt = points;
	ParamFunc[data->param_type](xt, yt);
	std::vector<Ubpa::pointf2> x;
	std::vector<Ubpa::pointf2> y;
	if (with_slope)
	{
		if (!edit_flag)
		{
			SlopeSpline(x, xt, data->xk);
			SlopeSpline(y, yt, data->yk);
		}
		else
		{
			std::vector<Slope> xk = data->xk;
			std::vector<Slope> yk = data->yk;
			for (int i = 0; i < data->points.size() - 1; i++)
			{
				const ImVec2 p1(origin.x + points[i][0], origin.y + points[i][1]);
				const ImVec2 p2(origin.x + rtangent[i][0], origin.y + rtangent[i][1]);
				xk[i].r = rtangent[i][0] - points[i][0];
				xk[i].r /= data->tangent_ratio[i].r;
				yk[i].r = rtangent[i][1] - points[i][1];
				yk[i].r /= data->tangent_ratio[i].r;
			}
			for (int i = data->points.size() - 1; i > 0; i--)
			{
				const ImVec2 p1(origin.x + points[i][0], origin.y + points[i][1]);
				const ImVec2 p2(origin.x + ltangent[i][0], origin.y + ltangent[i][1]);
				xk[i].l = points[i][0] - ltangent[i][0];
				xk[i].l /= data->tangent_ratio[i].l;
				yk[i].l = points[i][1] - ltangent[i][1];
				yk[i].l /= data->tangent_ratio[i].l;
			}
			SlopeSpline(x, xt, xk);
			SlopeSpline(y, yt, yk);
		}
	}
	else
	{
		CubicSpline(x, xt, data->xk, edit_flag);
		CubicSpline(y, yt, data->yk, edit_flag);
		if (!edit_flag)
		{
			for (int i = 0; i < points.size() - 1; i++)
			{
				Ubpa::pointf2 temp = Ubpa::pointf2(data->xk[i].r, data->yk[i].r);
				data->tangent_ratio[i].r = base_tangent_len / temp.distance(Ubpa::pointf2(0.f, 0.f));
				data->rtangent[i] = Ubpa::pointf2(data->points[i][0] + data->xk[i].r * data->tangent_ratio[i].r,
					data->points[i][1] + data->yk[i].r * data->tangent_ratio[i].r);
			}
			for (int i = points.size() - 1; i > 0; i--)
			{
				Ubpa::pointf2 temp = Ubpa::pointf2(data->xk[i].l, data->yk[i].l);
				data->tangent_ratio[i].l = base_tangent_len / temp.distance(Ubpa::pointf2(0.f, 0.f));
				data->ltangent[i] = Ubpa::pointf2(data->points[i][0] - data->xk[i].l * data->tangent_ratio[i].l,
					data->points[i][1] - data->yk[i].l * data->tangent_ratio[i].l);
			}
		}
	}
	std::vector<Ubpa::pointf2> p = y;
	for (int i = 0; i < p.size(); i++)
		p[i][0] = x[i][1];
	// draw tangent lines and points
	if (data->edit_point == 2 || data->edit_point == 3)
	{
		for (int i = 0; i < data->points.size() - 1; i++)
		{
			const ImVec2 p1(origin.x + points[i][0], origin.y + points[i][1]);
			const ImVec2 p2(origin.x + rtangent[i][0], origin.y + rtangent[i][1]);
			if (!edit_flag)
			{
				data->xk[i].r = rtangent[i][0] - points[i][0];
				data->xk[i].r /= data->tangent_ratio[i].r;
				data->yk[i].r = rtangent[i][1] - points[i][1];
				data->yk[i].r /= data->tangent_ratio[i].r;
				draw_list->AddLine(p1, p2, slope_col, 2.f);
				draw_list->AddCircleFilled(p2, point_radius, normal_point_col);
			}
			else
			{
				if (edit_flag == 2)
					draw_list->AddLine(p1, p2, select_slope_col, 2.f);
				draw_list->AddCircleFilled(p2, point_radius, select_point_col);
			}
		}
		for (int i = data->points.size() - 1; i > 0; i--)
		{
			const ImVec2 p1(origin.x + points[i][0], origin.y + points[i][1]);
			const ImVec2 p2(origin.x + ltangent[i][0], origin.y + ltangent[i][1]);
			if (!edit_flag)
			{
				data->xk[i].l = points[i][0] - ltangent[i][0];
				data->xk[i].l /= data->tangent_ratio[i].l;
				data->yk[i].l = points[i][1] - ltangent[i][1];
				data->yk[i].l /= data->tangent_ratio[i].l;
				draw_list->AddLine(p1, p2, slope_col, 2.f);
				draw_list->AddCircleFilled(p2, point_radius, normal_point_col);
			}
			else
			{
				if (edit_flag == 2)
					draw_list->AddLine(p1, p2, select_slope_col, 2.f);
				draw_list->AddCircleFilled(p2, point_radius, select_point_col);
			}
		}
	}
	// draw spline
	for (int i = 1; i < p.size(); i++)
	{
		AddLine(ImVec2(origin.x + p[i - 1][0], origin.y + p[i - 1][1]),
			ImVec2(origin.x + p[i][0], origin.y + p[i][1]), draw_list, edit_flag);
	}
}

inline float h0(const float x0, const float x1, const float x)
{
	return (1.f + 2.f * (x - x0) / (x1 - x0)) * ((x - x1) * (x - x1)) / ((x0 - x1) * (x0 - x1));
}
inline float h1(const float x0, const float x1, const float x)
{
	return h0(x1, x0, x);
}
inline float H0(const float x0, const float x1, const float x)
{
	return (x - x0) * ((x - x1) * (x - x1)) / ((x0 - x1) * (x0 - x1));
}
inline float H1(const float x0, const float x1, const float x)
{
	return H0(x1, x0, x);
}

void SlopeSpline(std::vector<Ubpa::pointf2>& ret, std::vector<Ubpa::pointf2>& p, std::vector<Slope>& k)
{
	//spdlog::info("SlopeSpline");
	for (int i = 0; i < p.size() - 1; i++)
	{
		float y0 = p[i][1];
		float y1 = p[i + 1][1];
		float dy0 = k[i].r;
		float dy1 = k[i + 1].l;
		for (float x = p[i][0]; x <= p[i + 1][0]; x += t_step)
		{
			//S = h0*y0 + h1*y1 + dy0*H0 + dy1*H1;
			ret.push_back(Ubpa::pointf2(x,
				y0 * h0(p[i][0], p[i + 1][0], x) +
				y1 * h1(p[i][0], p[i + 1][0], x) +
				dy0 * H0(p[i][0], p[i + 1][0], x) +
				dy1 * H1(p[i][0], p[i + 1][0], x)));
		}
	}
}

void CubicSpline(std::vector<Ubpa::pointf2>& ret, std::vector<Ubpa::pointf2>& p, std::vector<Slope>& k, bool edit_flag)
{
	// if (!edit_flag && k.size() < p.size()) k.resize(p.size());
	size_t n = p.size() - 1;
	std::vector<float> u(n);
	std::vector<float> h(n);
	std::vector<float> v(n);
	std::vector<float> b(n);
	h[0] = p[1][0] - p[0][0];
	b[0] = 6.f * (p[1][1] - p[0][1]) / h[0];
	for (int i = 1; i < n; i++)
	{
		h[i] = p[i + 1][0] - p[i][0];
		u[i] = 2.f * (h[i] + h[i - 1]);
		b[i] = 6.f * (p[i + 1][1] - p[i][1]) / h[i];
		v[i] = b[i] - b[i - 1];
	}
	std::vector<float> M(n + 1, 0.f);
	for (int i = 2; i < n; i++)
	{
		b[i] = h[i - 1] / u[i - 1];
		u[i] -= h[i - 1] * b[i];
	}
	// Ly=V
	for (int i = 2; i < n; i++)
	{
		v[i] -= b[i] * v[i - 1];
	}
	// UM=y
	if (n > 1)
		M[n - 1] = v[n - 1] / u[n - 1];
	for (int i = n - 2; i >= 1; i--)
	{
		M[i] = (v[i] - h[i] * M[i + 1]) / u[i];
	}
	for (int i = 0; i < n; i++)
	{
		/*float a1 = M[i] / (6.f * h[i]);
		float a2 = M[i + 1] / (6 * h[i]);
		float a3 = (p[i + 1][1] / h[i] - M[i + 1] * h[i] / 6.f);
		float a4 = (p[i][1] / h[i] - M[i] * h[i] / 6.f);*/
		float y0 = p[i][1];
		float y1 = p[i + 1][1];
		float dy0 = (-h[i]) * (M[i] * 2 + M[i + 1]) / 6.f + (p[i + 1][1] - p[i][1]) / h[i];
		float dy1 = (h[i]) * (M[i + 1] * 2 + M[i]) / 6.f + (p[i + 1][1] - p[i][1]) / h[i];
		if (!edit_flag)
		{
			k[i].r = dy0;
			// spdlog::info("{:.5f}", k[i].r);
			if (i != 0)
				k[i].l = dy0;
			if (i == n - 1)
				k[i + 1].l = dy1;
		}
		for (float x = p[i][0]; x <= p[i + 1][0]; x += t_step)
		{
			//S = h0*y0 + h1*y1 + dy0*H0 + dy1*H1;
			ret.push_back(Ubpa::pointf2(x,
				y0 * h0(p[i][0], p[i + 1][0], x) +
				y1 * h1(p[i][0], p[i + 1][0], x) +
				dy0 * H0(p[i][0], p[i + 1][0], x) +
				dy1 * H1(p[i][0], p[i + 1][0], x)));
		}
	}
}

// xt, yt == data->points at first
void UniformParameterize(std::vector<Ubpa::pointf2>& xt, std::vector<Ubpa::pointf2>& yt)
{
	//spdlog::info("UniformParameterize");
	// t is in the range 0 to 1, and uniformed mapping to x and y
	float step = 1.f / (xt.size() - 1), t = 0.0;
	for (int i = 0; i < xt.size(); t += step, i++)
	{
		// (t_i, x_i)
		xt[i][1] = xt[i][0];
		xt[i][0] = t;
		// (t_i, y_i)
		yt[i][0] = t;
		//spdlog::info("{:.2f}", t);
	}
}

void ChordalParameterize(std::vector<Ubpa::pointf2>& xt, std::vector<Ubpa::pointf2>& yt)
{
	//spdlog::info("ChordalParameterize");
	std::vector<float> k(xt.size() - 1);
	float sum = 0.f, t = 0.f;
	for (int i = 0; i < xt.size() - 1; i++)
	{
		k[i] = xt[i + 1].distance(xt[i]);
		sum += k[i];
	}
	for (int i = 0; i < xt.size(); i++)
	{
		// (t_i, x_i)
		xt[i][1] = xt[i][0];
		xt[i][0] = t / sum;
		// (t_i, y_i)
		yt[i][0] = t / sum;
		//spdlog::info("{:.2f}", t/sum);
		if (i < k.size())
			t += k[i];
	}
}

void CentripetalParameterize(std::vector<Ubpa::pointf2>& xt, std::vector<Ubpa::pointf2>& yt)
{
	//spdlog::info("CentripetalParameterize");
	std::vector<float> k(xt.size() - 1);
	float sum = 0.f, t = 0.f;
	for (int i = 0; i < xt.size() - 1; i++)
	{
		k[i] = std::sqrt(xt[i + 1].distance(xt[i]));
		sum += k[i];
	}
	for (int i = 0; i < xt.size(); i++)
	{
		// (t_i, x_i)
		xt[i][1] = xt[i][0];
		xt[i][0] = t / sum;
		// (t_i, y_i)
		yt[i][0] = t / sum;
		//spdlog::info("{:.2f}", t / sum);
		if (i < k.size())
			t += k[i];
	}
}

// get angle ??ABC
inline float getAngle(Ubpa::pointf2& A, Ubpa::pointf2& B, Ubpa::pointf2& C)
{
	Ubpa::vecf2 BA = A - B;
	Ubpa::vecf2 BC = C - B;
	return acos(BA.dot(BC) / (BA.norm() * BC.norm()));
}

inline float getDeltaT(float d0, float d1, float d2, float a0, float a1, int flag = 1)
{
	//return d0 * (1.f + 1.5f * d1 * (std::min(Ubpa::PI<float> / 2.f, Ubpa::PI<float> -a)) / (d0 + d1));
	return d1 * (1.f +
		1.5f * d0 * (std::min(Ubpa::PI<float> / 2.f, Ubpa::PI<float> -a0) / (d0 + d1)) +
		1.5f * d2 * (std::min(Ubpa::PI<float> / 2.f, Ubpa::PI<float> -a1) / (d1 + d2)) * flag);
}

void FoleyParameterize(std::vector<Ubpa::pointf2>& xt, std::vector<Ubpa::pointf2>& yt)
{
	//spdlog::info("FoleyParameterize");
	// at least 3 points
	if (xt.size() < 3)
	{
		UniformParameterize(xt, yt);
		return;
	}
	std::vector<float> d(xt.size() - 1);
	std::vector<float> a(xt.size() - 2);
	std::vector<float> dt(xt.size() - 1);
	for (int i = 0; i < xt.size() - 1; i++)
	{
		d[i] = xt[i + 1].distance(xt[i]);
	}
	for (int i = 1; i < xt.size() - 1; i++)
	{
		a[i - 1] = getAngle(xt[i - 1], xt[i], xt[i + 1]);
	}
	dt[0] = getDeltaT(d[1], d[0], 0.f, a[0], 0.f, 0);
	int n = dt.size() - 1;
	if (n > 0)
		dt[n] = getDeltaT(d[n - 1], d[n], 0.f, a[n - 1], 0.f, 0);
	for (int i = 1; i < n; i++)
		dt[i] = getDeltaT(d[i - 1], d[i], d[i + 1], a[i - 1], a[i]);
	float sum = 0.f, t = 0.f;
	for (int i = 0; i <= n; i++)
		sum += dt[i];
	for (int i = 0; i < xt.size(); i++)
	{
		// (t_i, x_i)
		xt[i][1] = xt[i][0];
		xt[i][0] = t / sum;
		// (t_i, y_i)
		yt[i][0] = t / sum;
		//spdlog::info("{:.2f}", t / sum);
		if (i < dt.size())
			t += dt[i];
	}
}