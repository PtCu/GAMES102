#include "CanvasSystem.h"

#include "../Components/CanvasData.h"

#include <_deps/imgui/imgui.h>

#include<Eigen/Dense>
#include<Eigen/Core>
#include <Eigen/Sparse>

using namespace Ubpa;
using namespace Eigen;


static void DrawLine(CanvasData*, ImDrawList*, const ImVec2&);


static void DrawPoints(CanvasData*, ImDrawList*, const ImVec2&);


// 重载加法
Ubpa::pointf2 operator+(const Ubpa::pointf2& a, const Ubpa::pointf2& b) {
	Ubpa::pointf2 pf2;
	pf2[0] = a[0] + b[0];
	pf2[1] = a[1] + b[1];
	return pf2;
}

// 重载减法
Ubpa::pointf2 operator-(const Ubpa::pointf2& a, const Ubpa::pointf2& b) {
	Ubpa::pointf2 pf2;
	pf2[0] = a[0] - b[0];
	pf2[1] = a[1] - b[1];
	return pf2;
}

// 重载乘法
Ubpa::pointf2 operator*(const Ubpa::pointf2& a, const float b) {
	Ubpa::pointf2 pf2;
	pf2[0] = a[0] * b;
	pf2[1] = a[1] * b;
	return pf2;
}

// 重载除法
Ubpa::pointf2 operator/(const Ubpa::pointf2& a, const float b) {
	Ubpa::pointf2 pf2;
	pf2[0] = a[0] / b;
	pf2[1] = a[1] / b;
	return pf2;
}
std::vector<Ubpa::pointf2> Chaikin_subdivision(std::vector<Ubpa::pointf2>&p,  bool close) {
	std::vector<Ubpa::pointf2> output;
	output.clear();
	int n = p.size();
	if (close) {
		for (int i = 0; i < n; ++i) {
			output.push_back(p[(i - 1 + n) % n] * 0.25f + p[i] * 0.75f);
			output.push_back(p[i] * 0.75f + p[(i + 1) % n] * 0.25f);
		}
	}
	if (!close) {
		output.push_back(p[0] * 0.75f + p[1] * 0.25f);
		for (int i = 1; i < n-1; ++i) {
			output.push_back(p[(i - 1 + n) % n] * 0.25f + p[i] * 0.75f);
			output.push_back(p[i] * 0.75f + p[(i + 1) % n] * 0.25f);
		}
		output.push_back(p[(n - 1 - 1 + n) % n] * 0.25f + p[n - 1] * 0.75f);
	}
	return output;
}




std::vector<Ubpa::pointf2> cubic_subdivision(std::vector<Ubpa::pointf2>& p, bool close) {
	std::vector<Ubpa::pointf2> output;
	output.clear();
	int n = p.size();
	if (close) {
		for (int i = 0; i < n; ++i) {
			output.push_back(p[(i - 1 + n) % n] * 0.125f + p[i] * 0.75f + p[(i + 1) % n] * 0.125f);
			output.push_back(p[i] * 0.5f + p[(i + 1) % n] * 0.5f);
		}
	}
	if (!close) {
		output.push_back(p[0] * 0.5f + p[1] * 0.5f);
		for (int i = 1; i < n - 1; ++i) {
			output.push_back(p[(i - 1 + n) % n] * 0.125f + p[i]*0.75f + p[(i + 1) % n] * 0.125f);
			output.push_back(p[i] * 0.5f + p[(i + 1) % n] * 0.5f);
		}
	}
	return output;
}

/*
/// @brief      四点插值型细分曲线
/// @details
/// @param[in]  :
/// @return
/// @attention
*/
std::vector<Ubpa::pointf2> quad_subdivision(std::vector<Ubpa::pointf2>& p, bool close, float alpha = 0.075) {
	std::vector<Ubpa::pointf2> output;
	output.clear();
	int n = p.size();
	if (close) {
		for (int i = 0; i < n; ++i) {
			output.push_back(p[i]);
			output.push_back((p[i] + p[(i + 1) % n]) / 2.0f + ((p[i] + p[(i + 1) % n]) / 2.0f - (p[(i - 1 + n) % n] + p[(i + 2) % n]) / 2.0f) * alpha);
		}
	}
	if (!close) {
		for (int i = 0; i < n - 1; ++i) {
			output.push_back(p[i]);
			output.push_back((p[i] + p[(i + 1) % n]) / 2.0f + ((p[i] + p[(i + 1) % n]) / 2.0f - (p[(i - 1 + n) % n] + p[(i + 2) % n]) / 2.0f) * alpha);
		}

	}
	return output;
}




void CanvasSystem::OnUpdate(Ubpa::UECS::Schedule& schedule) {
	schedule.RegisterCommand([](Ubpa::UECS::World* w) {
		auto data = w->entityMngr.GetSingleton<CanvasData>();
		if (!data)
			return;

		if (ImGui::Begin("Canvas")) {
		
			ImGui::Text("When adding new points, press left key to add new points and press middle key to stop add. \n Then press right key to edit or add new");

			if (ImGui::RadioButton("Chaikin ", data->narrow_type == Narrow_Type::chaikin))
				data->narrow_type = Narrow_Type::chaikin;
			if (ImGui::RadioButton("Cubic ", data->narrow_type == Narrow_Type::cubic))
				data->narrow_type = Narrow_Type::cubic;
			if (ImGui::RadioButton("Quad ", data->narrow_type == Narrow_Type::quad))
				data->narrow_type = Narrow_Type::quad;
			if (ImGui::RadioButton("Close ", data->isOpen == true))
				data->isOpen = true;

			ImGui::SliderInt("times", &data->times, 10, 1000);

			// Typically you would use a BeginChild()/EndChild() pair to benefit from a clipping region + own scrolling.
			// Here we demonstrate that this can be replaced by simple offsetting + custom drawing + PushClipRect/PopClipRect() calls.
			// To use a child window instead we could use, e.g:
			//      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));      // Disable padding
			//      ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(50, 50, 50, 255));  // Set a background color
			//      ImGui::BeginChild("canvas", ImVec2(0.0f, 0.0f), true, ImGuiWindowFlags_NoMove);
			//      ImGui::PopStyleColor();
			//      ImGui::PopStyleVar();
			//      [...]
			//      ImGui::EndChild();

			// Using InvisibleButton() as a convenience 1) it will advance the layout cursor and 2) allows us to use IsItemHovered()/IsItemActive()
			ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();      // ImDrawList API uses screen coordinates!
			ImVec2 canvas_sz = ImGui::GetContentRegionAvail();   // Resize canvas to what's available
			if (canvas_sz.x < 50.0f) canvas_sz.x = 50.0f;
			if (canvas_sz.y < 50.0f) canvas_sz.y = 50.0f;
			ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);

			// Draw border and background color
			ImGuiIO& io = ImGui::GetIO();
			ImDrawList* draw_list = ImGui::GetWindowDrawList();
			draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(50, 50, 50, 255));
			draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

			// This will catch our interactions
			ImGui::InvisibleButton("canvas", canvas_sz, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
			const bool is_hovered = ImGui::IsItemHovered(); // Hovered
			const bool is_active = ImGui::IsItemActive();   // Held
			const ImVec2 origin(canvas_p0.x + data->scrolling[0], canvas_p0.y + data->scrolling[1]); // Lock scrolled origin
			const pointf2 mouse_pos_in_canvas(io.MousePos.x - origin.x, io.MousePos.y - origin.y);

			// Pan (we use a zero mouse threshold when there's no context menu)
			// You may decide to make that threshold dynamic based on whether the mouse is hovering something etc.
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
				if (data->generate_line)
					data->points.resize(data->points.size() - 2);
				data->generate_line = false;
				if (ImGui::MenuItem("Remove one", NULL, false, data->points.size() > 0)) { data->points.resize(data->points.size() - 1); }
				if (ImGui::MenuItem("Remove all", NULL, false, data->points.size() > 0)) { data->points.clear(); }
				ImGui::EndPopup();
			}
			if (data->cur_times != data->times) {
				data->cur_times = data->times;
				data->generate_line = true;
			}
			if (data->cur_narrow_type != data->narrow_type) {
				data->cur_narrow_type = data->narrow_type;
				data->generate_line = true;
			}
			// Draw grid + all lines in the canvas
			draw_list->PushClipRect(canvas_p0, canvas_p1, true);
			if (data->opt_enable_grid)
			{
				const float GRID_STEP = 64.0f;
				for (float x = fmodf(data->scrolling[0], GRID_STEP); x < canvas_sz.x; x += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x + x, canvas_p0.y), ImVec2(canvas_p0.x + x, canvas_p1.y), IM_COL32(200, 200, 200, 40));
				for (float y = fmodf(data->scrolling[1], GRID_STEP); y < canvas_sz.y; y += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x, canvas_p0.y + y), ImVec2(canvas_p1.x, canvas_p0.y + y), IM_COL32(200, 200, 200, 40));
			}

			if (is_active && !data->generate_line && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
				data->points.push_back(mouse_pos_in_canvas);
				data->generate_line = true;
			}
			DrawPoints(data, draw_list, origin);
			if (data->generate_line && data->points.size() > 3) {

				if (data->narrow_type == Narrow_Type::chaikin) {
					data->output_points.clear();
					data->output_points=Chaikin_subdivision(data->points, data->isOpen);
					for (int i = 0; i < data->times; ++i) {
						data->output_points = Chaikin_subdivision(data->output_points, data->isOpen);
					}
					
				}
				else if (data->narrow_type == Narrow_Type::cubic) {
					data->output_points.clear();
					data->output_points = cubic_subdivision(data->points, data->isOpen);
					for (int i = 0; i < data->times; ++i) {
						data->output_points = cubic_subdivision(data->points, data->isOpen);
					}
				}
				else if (data->narrow_type == Narrow_Type::quad) {
					data->output_points.clear();
					data->output_points = quad_subdivision(data->points, data->isOpen);
					for (int i = 0; i < data->times; ++i) {
						data->output_points = quad_subdivision(data->points, data->isOpen);
					}
				}

			}
			data->generate_line = false;
			if(data->points.size()>3)
			DrawLine(data, draw_list, origin);
			

			//// Draw grid + all lines in the canvas
			draw_list->PushClipRect(canvas_p0, canvas_p1, true);
		
		}

		ImGui::End();
		});
}


void DrawPoints(CanvasData* data, ImDrawList* draw_list, const ImVec2& origin) {
	for (size_t i = 0; i < data->points.size(); ++i) {
		draw_list->AddCircleFilled(ImVec2(origin.x + data->points[i][0], origin.y + data->points[i][1]), 5.0f, IM_COL32(255, 0, 0, 255));
	}
}



void DrawLine(CanvasData* data, ImDrawList* draw_list, const ImVec2& origin) {

	for (int i = 0; i < data->output_points.size() - 1; ++i) {
		draw_list->AddLine(ImVec2(origin.x + data->output_points[i][0], origin.y + data->output_points[i][1]), ImVec2(origin.x + data->output_points[i + 1][0], origin.y + data->output_points[i + 1][1]), IM_COL32(0, 255, 255, 255));
	}
}

