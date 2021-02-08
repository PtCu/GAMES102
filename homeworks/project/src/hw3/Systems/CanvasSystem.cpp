#include "CanvasSystem.h"

#include "../Components/CanvasData.h"

#include <_deps/imgui/imgui.h>


#include <Dense>

using namespace Ubpa;



void LSM(std::vector<float>& sample_x, std::vector<float>& sample_y, std::vector<float>& output_y, int num, float lambda, int m, float b0, float sigma) {
	//Using LSM
	if (sample_x.size() < 3) return;
	if (m + 1 > sample_x.size()) m = sample_x.size() - 1;
	Eigen::MatrixXf A(sample_x.size(), m + 1);
	Eigen::VectorXf as(m + 1);
	Eigen::VectorXf y(sample_x.size());
	for (int i = 0; i < sample_x.size(); ++i) {
		for (int j = 0; j < m + 1; ++j) {
			A(i, j) = std::exp(-0.5 * std::powf((sample_x[i] - sample_x[j]), 2) / sigma);
		}
		y(i) = sample_y[i] - b0;
	}
	as = (A.transpose() * A).inverse() * A.transpose() * y;

	for (int i = 0; i < num; ++i) {
		auto x = i * 1.0f / num;
		float y = b0;
		for (size_t i = 0; i < m + 1; ++i) {
			y += as(i) * std::exp(-0.5 * std::powf((x - sample_x[i]), 2) / sigma);
		}
		output_y.push_back(y);
	}

	
}
void Guass_interpolation(std::vector<float>& sample_x, std::vector<float>& sample_y, std::vector<float>& output_y, int num, float lambda, int m, float b0, float sigma) {
	//Using Guass
	if (sample_x.size() < 2) return;
	Eigen::MatrixXf G(sample_x.size(), sample_x.size());
	Eigen::VectorXf bs(sample_x.size());
	Eigen::VectorXf y(sample_x.size());
	for (size_t i = 0; i < sample_x.size(); ++i) {
		for (size_t j = 0; j < sample_x.size(); ++j) {
			G(i, j) = std::exp(-0.5 * std::powf((sample_x[i] - sample_x[j]), 2) / sigma);
		}
		y(i) = sample_y[i] - b0;
	}
	bs = G.colPivHouseholderQr().solve(y);

	for (int i = 0; i < num;++i) {
		auto x = i * 1.0f / num;
		float y = b0;
		for (size_t i = 0; i < sample_x.size(); ++i) {
			y += bs(i) * std::exp(-0.5 * std::powf(x - sample_x[i], 2) / sigma);
		}
		output_y.push_back(y);
	}
}

void RR(std::vector<float>& sample_x, std::vector<float>& sample_y, std::vector<float>&output_y, int num,float lambda, int m, float b0, float sigma) {
	//Using RR
	if (sample_x.size() < 3) return;
	if (m + 1 > sample_x.size()) m = sample_x.size() - 1;
	Eigen::MatrixXf A(sample_x.size(), m + 1);
	Eigen::VectorXf as(m + 1);
	Eigen::VectorXf y(sample_x.size());
	for (int i = 0; i < sample_x.size(); ++i) {
		for (int j = 0; j < m + 1; ++j) {
			A(i, j) = std::exp(-0.5 * std::powf((sample_x[i] - sample_x[j]), 2) / sigma);
		}
		y(i) = sample_y[i] - b0;
	}
	as = (A.transpose() * A + lambda * Eigen::MatrixXf::Identity(m + 1, m + 1)).inverse() * A.transpose() * y;

	for (int i = 0; i < num;++i) {
		auto x = i * 1.0f / num;
		float y = b0;
		for (size_t i = 0; i < m + 1; ++i) {
			y += as(i) * std::exp(-0.5 * std::powf((x - sample_x[i]), 2) / sigma);
		}
		output_y.push_back(y);
	}
}


void 
Curve_parameterize
(	std::vector<Ubpa::pointf2>& input_points, 
	std::vector<Ubpa::pointf2>& output_points,
	const std::function<void(std::vector<float>&,std::vector<Ubpa::pointf2>&)>&param_f,
	const std::function<void(std::vector<float>& sample_x, std::vector<float>& sample_y, std::vector<float>& output_y, int num, float lambda, int m, float b0, float sigma)>fitting_f,
	float lambda, int m, float b0, float sigma)
{
	//parameterize t
	std::vector<float> t;
	param_f(t, input_points);

	std::vector<float> input_x(input_points.size());
	std::vector<float> input_y(input_points.size());
	for (int i = 0; i < input_x.size(); ++i) {
		input_x[i] = input_points[i][0];
		input_y[i] = input_points[i][1];
	}
	auto min_x = *std::min_element(input_x.begin(), input_x.end());
	auto max_x = *std::max_element(input_x.begin(), input_x.end());
	int num = (int)(max_x - min_x) + 1;
	//Get fitting outcome.
	std::vector<float> output_x, output_y;
	fitting_f(t, input_x, output_x, num,lambda, m, b0, sigma);
	fitting_f(t, input_y, output_y, num,lambda, m, b0, sigma);

	for (int i = 0; i < output_x.size(); ++i) {
		output_points.push_back({ output_x[i],output_y[i] });
	}
	

}

void Uniform(std::vector<float>&t, std::vector<Ubpa::pointf2>& input_points) {
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

void Centripetal(std::vector<float>& t, std::vector<Ubpa::pointf2>& input_points) {
	t.resize(input_points.size());
	t[0] = 0;
	for (int i = 1; i < input_points.size(); ++i) {
		t[i] = t[i - 1] +std::sqrt((input_points[i] - input_points[i - 1]).norm());
	}
	for (int i = 1; i < input_points.size(); ++i)
		t[i] /= t.back();
}
void  Foley(std::vector<float>&t, std::vector<Ubpa::pointf2>& input_points) {
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
void CanvasSystem::OnUpdate(Ubpa::UECS::Schedule& schedule) {
	schedule.RegisterCommand([](Ubpa::UECS::World* w) {
		auto data = w->entityMngr.GetSingleton<CanvasData>();
		if (!data)
			return;

		if (ImGui::Begin("Canvas")) {
			ImGui::Checkbox("Enable grid", &data->opt_enable_grid);
			ImGui::Checkbox("Enable context menu", &data->opt_enable_context_menu);
			ImGui::Text("Mouse Left: drag to add lines,\nMouse Right: drag to scroll, click for context menu.");

			//ImGui::Checkbox("Lagrange_interpolation", &data->opt_proc[0]);
			//ImGui::Checkbox("Guass_interpolation", &data->opt_proc[1]);
			//ImGui::Checkbox("RR_approximation", &data->opt_proc[2]); //Ridge Regression
			//ImGui::Checkbox("LSM_approximation", &data->opt_proc[3]); //Least Squares Method

			ImGui::SliderFloat("Lamda", &data->lambda, -1000.0f, 1000.0f);
			ImGui::SliderFloat("sigma", &data->sigma, 0.0f, 1000.0f);
			ImGui::SliderFloat("b0", &data->b0, 0.0f, 500.0f);
			ImGui::SliderInt("Highest Power", &data->m, 1, 100);

			ImGui::Checkbox("Unifrom", &data->opt_param[0]);
			ImGui::Checkbox("Chord", &data->opt_param[1]); //Ridge Regression
			ImGui::Checkbox("Centripetal", &data->opt_param[2]); //Least Squares Method
			ImGui::Checkbox("Foley", &data->opt_param[3]);


			// Typically you would use a BeginChild()/EndChild() pair to benefit from a clipping
			// region + own scrolling. Here we demonstrate that this can be replaced by simple
			// offsetting + custom drawing + PushClipRect/PopClipRect() calls. To use a child window
			// instead we could use, e.g: ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0,
			// 0)); // Disable padding ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(50, 50, 50,
			// 255)); // Set a background color ImGui::BeginChild("canvas", ImVec2(0.0f, 0.0f),
			// true, ImGuiWindowFlags_NoMove); ImGui::PopStyleColor(); ImGui::PopStyleVar(); [...] ImGui::EndChild();

			// Using InvisibleButton() as a convenience 1) it will advance the layout cursor and 2)
			// allows us to use IsItemHovered()/IsItemActive()
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

			//// Add first and second point
			//if (is_hovered && !data->generate_line && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
			//{
			//	data->points.push_back(mouse_pos_in_canvas);
			//	data->points.push_back(mouse_pos_in_canvas);
			//	data->generate_line = true;
			//}
			//if (data->generate_line)
			//{
			//	data->points.back() = mouse_pos_in_canvas;
			//	if (!ImGui::IsMouseDown(ImGuiMouseButton_Left))
			//		data->generate_line = false;
			//}


			//Once clicked, generate relevant lines.
			if (is_active && !data->generate_line && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
				data->input_points.push_back(mouse_pos_in_canvas);
				data->generate_line = true;
			}
			if ( data->cur_m != data->m) {
				data->cur_m = data->m;
				data->generate_line = true;
			}
			if ( data->cur_b0 != data->b0) {
				data->cur_b0 = data->b0;
				data->generate_line = true;
			}
			if ( data->cur_lambda != data->lambda) {
				data->cur_lambda = data->lambda;
				data->generate_line = true;
			}
			if ( data->cur_sigma != data->sigma) {
				data->cur_sigma = data->cur_sigma;
				data->generate_line = true;
			}
			// Pan (we use a zero mouse threshold when there's no context menu) You may decide to
			// make that threshold dynamic based on whether the mouse is hovering something etc.

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
					data->input_points.resize(data->input_points.size() - 2);
				data->generate_line = false;
				if (ImGui::MenuItem("Remove one", NULL, false, data->input_points.size() > 0)) { data->input_points.resize(data->input_points.size() - 1); }
				if (ImGui::MenuItem("Remove all", NULL, false, data->input_points.size() > 0)) { data->input_points.clear(); }
				ImGui::EndPopup();
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
			for (size_t i = 0; i < data->input_points.size(); ++i) {
				draw_list->AddCircleFilled(ImVec2(origin.x + data->input_points[i][0], origin.y + data->input_points[i][1]), 5.0f, IM_COL32(255, 0, 0, 255));
			}

			constexpr float step = 1.0f;
			
			if (data->generate_line&& data->input_points.size() > 3) {
				data->fitting_f = Guass_interpolation;
				data->output_points.clear();
				if (data->opt_param[0]) {
					data->param_f = Uniform;
				}
				else if (data->opt_param[1]) {
					data->param_f = Chord;
				}
				else if (data->opt_param[2]) {
					data->param_f = Centripetal;
				}
				else if (data->opt_param[3]) {
					data->param_f = Foley;

				}

				Curve_parameterize(data->input_points, data->output_points, data->param_f, data->fitting_f, data->lambda, data->m, data->b0, data->sigma);
				
				
			}
			data->generate_line = false;
			if(data->input_points.size()>3)
			for (size_t i = 0; i < data->output_points.size() - 1; ++i) {
				draw_list->AddLine(ImVec2(origin.x + data->output_points[i][0], origin.y + data->output_points[i][1]), ImVec2(origin.x + data->output_points[i + 1][0], origin.y + data->output_points[i + 1][1]), IM_COL32(0, 255, 255, 255));
			}
			
		
			/*for (int n = 0; n < data->points.size(); n += 2)
				draw_list->AddLine(ImVec2(origin.x + data->points[n][0], origin.y + data->points[n][1]), ImVec2(origin.x + data->points[n + 1][0], origin.y + data->points[n + 1][1]), IM_COL32(255, 255, 0, 255), 2.0f);*/
			draw_list->PopClipRect();
		}

		ImGui::End();
		});
}
