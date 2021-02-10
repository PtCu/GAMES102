#include "CanvasSystem.h"

#include "../Components/CanvasData.h"

#include <_deps/imgui/imgui.h>

#include<Dense>
#include<Core>
#include <Sparse>

using namespace Ubpa;
using namespace Eigen;


constexpr int sample_num = 500;
constexpr float base_tangent_len = 50.f;
constexpr float t_step = 1.f / (sample_num - 1);
constexpr float point_radius = 3.f;
constexpr ImU32 line_col = IM_COL32(39, 117, 182, 255);
constexpr ImU32 edit_line_col = IM_COL32(39, 117, 182, 100);
constexpr ImU32 normal_point_col = IM_COL32(0, 255, 0, 50);
constexpr ImU32 select_point_col = IM_COL32(0, 0, 255, 100);
constexpr ImU32 slope_col = IM_COL32(122, 115, 116, 255);
constexpr ImU32 select_slope_col = IM_COL32(122, 115, 116, 100);

void Uniform(std::vector<float>& t, std::vector<Ubpa::pointf2>& input_points);
void Chord(std::vector<float>& t, std::vector<Ubpa::pointf2>& input_points);
void Centripetal(std::vector<float>& t, std::vector<Ubpa::pointf2>& input_points);
void Foley(std::vector<float>& t, std::vector<Ubpa::pointf2>& input_points);

static void DrawSlope(CanvasData*, ImDrawList*, const ImVec2&);

static void DrawLine(CanvasData*, ImDrawList*, const ImVec2&);

static void CubicSpline(std::vector<Ubpa::pointf2>&, std::vector<Ubpa::pointf2>&, std::vector<Slope>&);

static void DrawPoints(CanvasData*, ImDrawList*, const ImVec2&);

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

void SlopeSpline(std::vector<Ubpa::pointf2>& p, std::vector<Ubpa::pointf2>& ret, std::vector<Slope>& k)
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


//p is in the form of (ti,xi)
void CubicSpline(std::vector<pointf2>& p, std::vector<pointf2>& output, std::vector<Slope>& k) {
	int n = p.size() - 1; //节点数为n+1，方程数为n
	k.resize(n + 1);
	VectorXf v(n - 1);
	VectorXf h(n);
	VectorXf b(n);
	VectorXf u(n);
	h[0] = p[1][0] - p[0][0];
	b[0] = 6.f * (p[1][1] - p[0][1]) / h[0];
	u[0] = 0;
	for (int i = 1; i < n; ++i) {
		h[i] = p[i + 1][0] - p[i][0];
		u[i] = 2.f * (h[i] + h[i - 1]);
		b[i] = 6.f * (p[i + 1][1] - p[i][1]) / h[i];
		v[i - 1] = b[i] - b[i - 1];
	}
	VectorXf M_ = VectorXf::Zero(n - 1);
	if (p.size() == 3) {
		M_(0) = v(0) / u(1);
	}
	else if (p.size() > 3) {

		SparseMatrix<float> A(n - 1, n - 1);
		A.insert(0, 0) = u[1];
		A.insert(0, 1) = h[1];
		A.insert(n - 2, n - 2) = u[n - 1];
		A.insert(n - 2, n - 3) = h[n - 2];
		for (size_t i = 1; i < A.cols() - 1; ++i) {
			A.insert(i, i) = u[i + 1];
			A.insert(i, i - 1) = h[i];
			A.insert(i, i + 1) = h[i + 1];
		}
		A.makeCompressed();
		SparseLU<SparseMatrix<float>> solver;
		solver.compute(A);
		M_ = solver.solve(v);

	}
	VectorXf M = VectorXf::Zero(n + 1);
	for (size_t i = 0; i < n - 1; ++i) {
		M[i + 1] = M_[i];
	}
	for (int i = 0; i < n; i++) {
		float y0 = p[i][1];
		float y1 = p[i + 1][1];
		float dy0 = (-h[i]) * (M[i] * 2 + M[i + 1]) / 6.f + (p[i + 1][1] - p[i][1]) / h[i];
		float dy1 = (h[i]) * (M[i + 1] * 2 + M[i]) / 6.f + (p[i + 1][1] - p[i][1]) / h[i];

		k[i].r = dy0;
		// spdlog::info("{:.5f}", k[i].r);
		if (i != 0)
			k[i].l = dy0;
		if (i == n - 1)
			k[i + 1].l = dy1;

		for (float x = p[i][0]; x < p[i + 1][0]; x += t_step) {
			output.push_back({ x,
				y0 * h0(p[i][0], p[i + 1][0], x) +
				y1 * h1(p[i][0], p[i + 1][0], x) +
				dy0 * H0(p[i][0], p[i + 1][0], x) +
				dy1 * H1(p[i][0], p[i + 1][0], x) });
		}
	}



}

void CubicSpline(std::vector<pointf2>& p, std::vector<pointf2>& output) {
	int n = p.size() - 1; //节点数为n+1，方程数为n
	VectorXf v(n - 1);
	VectorXf h(n);
	VectorXf b(n);
	VectorXf u(n);
	h[0] = p[1][0] - p[0][0];
	b[0] = 6.f * (p[1][1] - p[0][1]) / h[0];
	u[0] = 0;
	for (int i = 1; i < n; ++i) {
		h[i] = p[i + 1][0] - p[i][0];
		u[i] = 2.f * (h[i] + h[i - 1]);
		b[i] = 6.f * (p[i + 1][1] - p[i][1]) / h[i];
		v[i - 1] = b[i] - b[i - 1];
	}
	VectorXf M_ = VectorXf::Zero(n - 1);
	if (p.size() == 3) {
		M_(0) = v(0) / u(1);
	}
	else if (p.size() > 3) {

		SparseMatrix<float> A(n - 1, n - 1);
		A.insert(0, 0) = u[1];
		A.insert(0, 1) = h[1];
		A.insert(n - 2, n - 2) = u[n - 1];
		A.insert(n - 2, n - 3) = h[n - 2];
		for (size_t i = 1; i < A.cols() - 1; ++i) {
			A.insert(i, i) = u[i + 1];
			A.insert(i, i - 1) = h[i];
			A.insert(i, i + 1) = h[i + 1];
		}
		A.makeCompressed();
		SparseLU<SparseMatrix<float>> solver;
		solver.compute(A);
		M_ = solver.solve(v);

	}
	VectorXf M = VectorXf::Zero(n + 1);
	for (size_t i = 0; i < n - 1; ++i) {
		M[i + 1] = M_[i];
	}
	for (int i = 0; i < n; i++) {
		float y0 = p[i][1];
		float y1 = p[i + 1][1];
		float dy0 = (-h[i]) * (M[i] * 2 + M[i + 1]) / 6.f + (p[i + 1][1] - p[i][1]) / h[i];
		float dy1 = (h[i]) * (M[i + 1] * 2 + M[i]) / 6.f + (p[i + 1][1] - p[i][1]) / h[i];

		for (float x = p[i][0]; x < p[i + 1][0]; x += t_step) {
			output.push_back({ x,
				y0 * h0(p[i][0], p[i + 1][0], x) +
				y1 * h1(p[i][0], p[i + 1][0], x) +
				dy0 * H0(p[i][0], p[i + 1][0], x) +
				dy1 * H1(p[i][0], p[i + 1][0], x) });
		}
	}



}




void CanvasSystem::OnUpdate(Ubpa::UECS::Schedule& schedule) {
	schedule.RegisterCommand([](Ubpa::UECS::World* w) {
		auto data = w->entityMngr.GetSingleton<CanvasData>();
		if (!data)
			return;

		if (ImGui::Begin("Canvas")) {
			ImGui::Checkbox("Enable grid", &data->opt_enable_grid);
			ImGui::Checkbox("Enable context menu", &data->opt_enable_context_menu);
			ImGui::Text("When adding new points, press left key to add new points and press middle key to stop add. \n Then press right key to edit or add new");

			if (ImGui::RadioButton("Cubic Spline ", data->fitting_type == 0))
				data->fitting_type = 0;
			ImGui::SameLine();
			if (ImGui::RadioButton("Cubic Bezier ", data->fitting_type == 1))
				data->fitting_type = 1;

			if (ImGui::RadioButton("C0 ", data->c_type == 0))
				data->c_type = 0;
			ImGui::SameLine();
			if (ImGui::RadioButton("C1 ", data->c_type == 1))
				data->c_type = 1;


			if (ImGui::RadioButton("Unifrom ", data->para_type == Para_Type::Uniform))
				data->para_type = Para_Type::Uniform;
			ImGui::SameLine();
			if (ImGui::RadioButton("Chord ", data->para_type == Para_Type::Chord))
				data->para_type = Para_Type::Chord;
			ImGui::SameLine();
			if (ImGui::RadioButton("Centripetal ", data->para_type == Para_Type::Centripetal))
				data->para_type = Para_Type::Centripetal;
			ImGui::SameLine();
			if (ImGui::RadioButton("Foley ", data->para_type == Para_Type::Foley))
				data->para_type = Para_Type::Foley;


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

			if (data->para_type == Para_Type::Uniform) {
				data->param_f = Uniform;
			}
			else if (data->para_type == Para_Type::Chord) {
				data->param_f = Chord;
			}
			else if (data->para_type == Para_Type::Centripetal) {
				data->param_f = Centripetal;
			}
			else if (data->para_type == Para_Type::Foley) {
				data->param_f = Foley;
			}


			//// Draw grid + all lines in the canvas
			draw_list->PushClipRect(canvas_p0, canvas_p1, true);
			if (data->opt_enable_grid)
			{
				const float GRID_STEP = 64.0f;
				for (float x = fmodf(data->scrolling[0], GRID_STEP); x < canvas_sz.x; x += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x + x, canvas_p0.y), ImVec2(canvas_p0.x + x, canvas_p1.y), IM_COL32(200, 200, 200, 40));
				for (float y = fmodf(data->scrolling[1], GRID_STEP); y < canvas_sz.y; y += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x, canvas_p0.y + y), ImVec2(canvas_p1.x, canvas_p0.y + y), IM_COL32(200, 200, 200, 40));
			}

			draw_list->PopClipRect();

			if (data->cur_state == State::adding_point) {
				if (is_hovered) {
					if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
						data->points.push_back(mouse_pos_in_canvas);

					}
					//新建顶点结束
					else if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
						if (!data->points.empty()) {
							data->points.pop_back();
						}
						data->last_state = State::adding_point;
						data->cur_state = State::done;

					}
					else if (data->last_state == State::init) {
						if (!data->points.empty()) {
							data->points.pop_back();
						}
						data->points.push_back(mouse_pos_in_canvas);
					}
					else if (data->last_state == State::done) {
						data->last_state = State::init;
						data->points.push_back(mouse_pos_in_canvas);
					}
				}
			}

			else if (data->cur_state == State::done) {
				//// Context menu (under default mouse threshold)
				ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
				if (data->opt_enable_context_menu && ImGui::IsMouseReleased(ImGuiMouseButton_Right) && drag_delta.x == 0.0f && drag_delta.y == 0.0f)
					ImGui::OpenPopupContextItem("context");
				if (ImGui::BeginPopup("context"))
				{
					if (ImGui::MenuItem("Remove one", NULL, false, data->points.size() > 0)) { data->points.resize(data->points.size() - 1); }
					if (ImGui::MenuItem("Remove all", NULL, false, data->points.size() > 0)) { data->points.clear(); }
					if (ImGui::MenuItem("Edit", NULL, false, data->points.size() > 0)) { data->cur_state = State::editing; data->last_state = State::done; }
					if (ImGui::MenuItem("Add new points", NULL, false, true)) { data->cur_state = State::adding_point; data->last_state = State::done; }
					ImGui::EndPopup();
				}
			}
			else if (data->cur_state == State::editing) {

				data->is_edit = true;
				for (int i = 0; i < data->points.size(); i++)
				{
					if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
						if (std::abs(data->points[i][0] - mouse_pos_in_canvas[0]) < point_radius && std::abs(data->points[i][1] - mouse_pos_in_canvas[1]) < point_radius)
						{
							data->editing_index = i;
							data->cur_edit_state = Edit_State::dragging_point;
							break;
						}
						if (std::abs(data->ltangent[i][0] - mouse_pos_in_canvas[0]) < point_radius && std::abs(data->ltangent[i][1] - mouse_pos_in_canvas[1]) < point_radius)
						{
							data->editing_tan_index = i + 1;
							data->cur_edit_state = Edit_State::dragging_tan_l;
							break;
						}
						if (std::abs(data->rtangent[i][0] - mouse_pos_in_canvas[0]) < point_radius && std::abs(data->rtangent[i][1] - mouse_pos_in_canvas[1]) < point_radius)
						{
							data->editing_tan_index = i + 1;
							data->cur_edit_state = Edit_State::dragging_tan_r;
							break;
						}
					}

				}
				if (data->cur_edit_state == Edit_State::dragging_point)
				{
					if (ImGui::IsMouseDown(ImGuiMouseButton_Left))
					{
						data->points[data->editing_index] = mouse_pos_in_canvas;
					}
					if (ImGui::IsMouseReleased(ImGuiMouseButton_Left))
					{
						data->cur_edit_state = Edit_State::init;
						data->last_edit_state = Edit_State::dragging_point;
					}
				}

				if (data->c_type == 0) {
					if (data->cur_edit_state == Edit_State::dragging_tan_l) {
						if (ImGui::IsMouseDown(ImGuiMouseButton_Left))
						{
							data->ltangent[data->editing_tan_index - 1] = mouse_pos_in_canvas;

						}
						if (ImGui::IsMouseReleased(ImGuiMouseButton_Left))
						{
							data->cur_edit_state = Edit_State::init;
							data->last_edit_state = Edit_State::dragging_tan_l;
						}
					}
					else if (data->cur_edit_state == Edit_State::dragging_tan_r) {
						if (ImGui::IsMouseDown(ImGuiMouseButton_Left))
						{
							data->rtangent[data->editing_tan_index - 1] = mouse_pos_in_canvas;
						}
						if (ImGui::IsMouseReleased(ImGuiMouseButton_Left))
						{
							data->cur_edit_state = Edit_State::init;
							data->last_edit_state = Edit_State::dragging_tan_r;
						}
					}
				}
				else if (data->c_type == 1) {
					if (data->cur_edit_state == Edit_State::dragging_tan_l) {
						float x, y;
						if (ImGui::IsMouseDown(ImGuiMouseButton_Left))
						{
							//相似三角形算距离
							data->ltangent[data->editing_tan_index - 1] = mouse_pos_in_canvas;

							float ratio_distance = data->points[data->editing_tan_index - 1].distance(data->rtangent[data->editing_tan_index - 1]) /
								data->points[data->editing_tan_index - 1].distance(mouse_pos_in_canvas);
							x = mouse_pos_in_canvas[0] - data->points[data->editing_tan_index - 1][0];
							x = data->points[data->editing_tan_index - 1][0] - x * ratio_distance;
							y = mouse_pos_in_canvas[1] - data->points[data->editing_tan_index - 1][1];
							y = data->points[data->editing_tan_index - 1][1] - y * ratio_distance;
							data->rtangent[data->editing_tan_index - 1] = Ubpa::pointf2(x, y);

						}
						if (ImGui::IsMouseReleased(ImGuiMouseButton_Left))
						{

							data->cur_edit_state = Edit_State::init;
							data->last_edit_state = Edit_State::dragging_tan_l;
						}
					}
					else if (data->cur_edit_state == Edit_State::dragging_tan_r) {
						float x, y;
						if (ImGui::IsMouseDown(ImGuiMouseButton_Left))
						{
							data->rtangent[data->editing_tan_index - 1] = mouse_pos_in_canvas;

							float ratio_distance = data->points[data->editing_tan_index - 1].distance(data->ltangent[data->editing_tan_index - 1]) /
								data->points[data->editing_tan_index - 1].distance(mouse_pos_in_canvas);
							x = mouse_pos_in_canvas[0] - data->points[data->editing_tan_index - 1][0];
							x = data->points[data->editing_tan_index - 1][0] - x * ratio_distance;
							y = mouse_pos_in_canvas[1] - data->points[data->editing_tan_index - 1][1];
							y = data->points[data->editing_tan_index - 1][1] - y * ratio_distance;
							data->ltangent[data->editing_tan_index - 1] = Ubpa::pointf2(x, y);

						}
						if (ImGui::IsMouseReleased(ImGuiMouseButton_Left))
						{

							data->cur_edit_state = Edit_State::init;
							data->last_edit_state = Edit_State::dragging_tan_r;
						}
					}

				}

				DrawSlope(data, draw_list, origin);

			}

			DrawPoints(data, draw_list, origin);

			DrawLine(data, draw_list, origin);




		}

		ImGui::End();
		});
}

void Uniform(std::vector<float>& t, std::vector<Ubpa::pointf2>& input_points) {
	t.resize(input_points.size());
	t[0] = 0;
	float dt = 1.0f / input_points.size();
	for (int i = 1; i < input_points.size(); ++i) {
		t[i] = t[i - 1] + dt;
	}
}
void Chord(std::vector<float>& t, std::vector<Ubpa::pointf2>& input_points) {
	t.resize(input_points.size());
	t[0] = 0;
	for (int i = 1; i < input_points.size(); ++i) {
		t[i] = t[i - 1] + (input_points[i] - input_points[i - 1]).norm();

	}
	for (int i = 1; i < input_points.size(); ++i)
		t[i] /= t.back();
}

void Centripetal(std::vector<float>& t, std::vector<Ubpa::pointf2>& input_points)
{
	t.resize(input_points.size());
	t[0] = 0;
	for (int i = 1; i < input_points.size(); ++i) {
		t[i] = t[i - 1] + std::sqrt((input_points[i] - input_points[i - 1]).norm());
	}
	for (int i = 1; i < input_points.size(); ++i)
		t[i] /= t.back();
}

void Foley(std::vector<float>& t, std::vector<Ubpa::pointf2>& input_points)
{
	/*t.resize(input_points.size());
	t[0] = 0;
	std::vector<float> a(input_points.size(),0.0);
	std::vector<float> k(input_points.size(),0.0);
	for (int i = 1; i < input_points.size(); ++i) {
		k[i] = (input_points[i] - input_points[i - 1]).norm();
	}
	for (int i = 1; i < input_points.size()-1; ++i) {
		a[i] = std::acos((input_points[i - 1] - input_points[i]).dot((input_points[i + 1] - input_points[i])) / (k[i] * k[i + 1]));
		a[i] = std::min(PI<float>-a[i], PI<float> / 2);
	}

	for (int i = 1; i < input_points.size()-1; ++i) {
		t[i] = t[i - 1] + (k[i] * (1.0f + 1.5f * a[i - 1] * k[i - 1] / (k[i] + k[i - 1]) + 1.5f * (a[i] * k[i] / (k[i + 1] + k[i]))));
	}
	for (int i = 1; i < input_points.size() ; ++i) {
		t[i] /= t.back();
	}*/
	t.resize(input_points.size());
	t[0] = 0;
	std::vector<float> a;
	int n = input_points.size();
	a.resize(n);
	std::vector<float> d(n, 0.0);
	a[0] = 0;
	for (size_t i = 1; i < n; ++i)
		d[i] = (input_points[i] - input_points[i - 1]).norm();
	for (size_t i = 1; i < n - 1; ++i)
	{
		float ai = std::acos((input_points[i] - input_points[i - 1]).normalize().dot(
			(input_points[i] - input_points[i + 1]).normalize()));
		a[i] = std::min(PI<float> -ai, PI<float> / 2);
	}
	for (size_t i = 1; i < n; ++i)
	{
		float b1, b2;
		b1 = i < n - 1 ? a[i] * d[i] / (d[i] + d[i + 1]) : 0.0f;
		b2 = i < n - 2 ? a[i + 1] * d[i + 1] / (d[i + 1] + d[i + 2]) : 0.0f;
		//b1 = i < n - 1 ? a[i] * d[i] / (d[i] + d[i + 1]) : a[i];
		//if (i < n - 2)
		//	b2 = a[i + 1] * d[i + 1] / (d[i + 1] + d[i + 2]);
		//else if (i == n - 2)
		//	b2 = a[i + 1];
		//else
		//	b2 = 0.0f;
		t[i] = t[i - 1] + d[i] * (1.0f + 1.5f * b1 + 1.5f * b2);
	}
	for (size_t i = 1; i < n; ++i)
		t[i] /= t.back();

}
void DrawPoints(CanvasData* data, ImDrawList* draw_list, const ImVec2& origin) {
	for (size_t i = 0; i < data->points.size(); ++i) {
		draw_list->AddCircleFilled(ImVec2(origin.x + data->points[i][0], origin.y + data->points[i][1]), 5.0f, IM_COL32(255, 0, 0, 255));
	}
}
void DrawLine(CanvasData* data, ImDrawList* draw_list, const ImVec2& origin) {
	if (data->points.size() == 2) {
		draw_list->AddLine(ImVec2(origin.x + data->points[0][0], origin.y + data->points[0][1]), ImVec2(origin.x + data->points[1][0], origin.y + data->points[1][1]), IM_COL32(0, 255, 255, 255));
	}
	else if (data->points.size() > 2) {
		std::vector<Ubpa::pointf2>& input_points = data->points;
		std::vector<float> t;
		data->param_f(t, input_points);

		std::vector<pointf2> tx(input_points.size()), ty(input_points.size());
		for (int i = 0; i < input_points.size(); ++i) {
			tx[i] = { t[i],input_points[i][0] };
			ty[i] = { t[i],input_points[i][1] };
		}
		std::vector<pointf2> outx, outy;
		data->rtangent.resize(data->points.size());
		data->ltangent.resize(data->points.size());
		data->tangent_ratio.resize(data->points.size());

		if (data->is_edit) {
			std::vector<Slope>& xk = data->xk;
			std::vector<Slope>& yk = data->yk;
			std::vector<Ubpa::pointf2>& points = data->points;
			std::vector<Ubpa::pointf2>& rtangent = data->rtangent;
			std::vector<Ubpa::pointf2>& ltangent = data->ltangent;
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
			SlopeSpline(tx, outx, data->xk);
			SlopeSpline(ty, outy, data->yk);

		}
		else {
			CubicSpline(tx, outx, data->xk);
			CubicSpline(ty, outy, data->yk);
			for (int i = 0; i < data->points.size() - 1; ++i) {
				data->tangent_ratio[i].r = base_tangent_len / Ubpa::pointf2(data->xk[i].r, data->yk[i].r).distance({ 0.f,0.f });
				data->rtangent[i] = Ubpa::pointf2(data->points[i][0] + data->xk[i].r * data->tangent_ratio[i].r,
					data->points[i][1] + data->yk[i].r * data->tangent_ratio[i].r);
			}
			for (int i = data->points.size() - 1; i > 0; i--)
			{
				data->tangent_ratio[i].l = base_tangent_len / Ubpa::pointf2(data->xk[i].l, data->yk[i].l).distance({ 0.f,0.f });
				data->ltangent[i] = Ubpa::pointf2(data->points[i][0] - data->xk[i].l * data->tangent_ratio[i].l,
					data->points[i][1] - data->yk[i].l * data->tangent_ratio[i].l);
			}

		}



		for (int i = 0; i < outx.size() - 1; i++)
		{
			draw_list->AddLine(ImVec2(origin.x + outx[i][1], origin.y + outy[i][1]), ImVec2(origin.x + outx[i + 1][1], origin.y + outy[i + 1][1]), IM_COL32(0, 255, 255, 255));
		}
	}
}

void DrawSlope(CanvasData* data, ImDrawList* draw_list, const ImVec2& origin) {
	if (data->points.size() <= 2) return;

	for (int i = 0; i < data->points.size() - 1; i++)
	{
		const ImVec2 p1(origin.x + data->points[i][0], origin.y + data->points[i][1]);
		const ImVec2 p2(origin.x + data->rtangent[i][0], origin.y + data->rtangent[i][1]);
		draw_list->AddLine(p1, p2, slope_col, 2.f);
		draw_list->AddCircleFilled(p2, point_radius, normal_point_col);

	}
	for (int i = data->points.size() - 1; i > 0; i--)
	{
		const ImVec2 p1(origin.x + data->points[i][0], origin.y + data->points[i][1]);
		const ImVec2 p2(origin.x + data->ltangent[i][0], origin.y + data->ltangent[i][1]);
		draw_list->AddLine(p1, p2, slope_col, 2.f);
		draw_list->AddCircleFilled(p2, point_radius, normal_point_col);

	}
}