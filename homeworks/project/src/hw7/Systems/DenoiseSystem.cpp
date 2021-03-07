//From 优秀作业参考 http://staff.ustc.edu.cn/~lgliu/Courses/GAMES102_2020/default.html#GAMES

#include "DenoiseSystem.h"
#include "../Components/DenoiseData.h"
#include <_deps/imgui/imgui.h>
#include <spdlog/spdlog.h>
#include <cstring>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Sparse>

#define EPSILON 1E-4F //取ZERO<float>有时会报错
#define PI 3.1415926
using namespace std;
using namespace Ubpa;

void MeshToHEMesh(DenoiseData* data);
void HEMeshToMesh(DenoiseData* data);
rgbf ColorMap(float c);
float getCotbetween(Vertex* adjV, Vertex* v);
void GetMinSurface(DenoiseData* data);

void circle_param(DenoiseData* data, vector<Vertex*>& boundary, Eigen::VectorXf& b_u, Eigen::VectorXf& b_v, float R);
void square_param(DenoiseData* data, vector<Vertex*>& boundary, Eigen::VectorXf& b_u, Eigen::VectorXf& b_v, float R);
void arc_param(DenoiseData* data, vector<Vertex*>& boundary, Eigen::VectorXf& b_u, Eigen::VectorXf& b_v, float R);

void Parameterization(DenoiseData* data);
void DenoiseSystem::OnUpdate(Ubpa::UECS::Schedule& schedule) {
	schedule.RegisterCommand([](Ubpa::UECS::World* w) {
		auto data = w->entityMngr.GetSingleton<DenoiseData>();
		if (!data)
			return;

		if (ImGui::Begin("Denoise")) {
			ImGui::Text("Operation:"); ImGui::SameLine();
			if (ImGui::Button("Generate min surface")) {
				[&]() {
					if (!data->changed) {
						data->copy = *data->mesh;
					}

					MeshToHEMesh(data);
					if (!data->heMesh->IsTriMesh()) {
						spdlog::warn("HEMesh isn't triangle mesh");
						return;
					}
					GetMinSurface(data);
					data->changed = true;
					HEMeshToMesh(data);
				}();
			}
			if (ImGui::Button("Parameterization")) {
				[&]() {
					if (!data->changed) {
						data->copy = *data->mesh;
					}
					MeshToHEMesh(data);
					if (!data->heMesh->IsTriMesh()) {
						spdlog::warn("HEMesh isn't triangle mesh");
						return;
					}
					Parameterization(data);
					data->changed = true;
					HEMeshToMesh(data);
				}();
			}
			ImGui::SameLine();
			if (ImGui::Button("Recover Mesh")) {
				[&]() {
					if (!data->mesh) {
						spdlog::warn("mesh is nullptr");
						return;
					}
					if (data->copy.GetPositions().empty()) {
						spdlog::warn("copied mesh is empty");
						return;
					}

					*data->mesh = data->copy;

					spdlog::info("recover success");
				}();
			}

			ImGui::Text("Visualization:"); ImGui::SameLine();
			if (ImGui::Button("Normal")) {
				[&]() {
					if (!data->mesh) {
						spdlog::warn("mesh is nullptr");
						return;
					}

					data->mesh->SetToEditable();
					const auto& normals = data->mesh->GetNormals();
					std::vector<rgbf> colors;
					for (const auto& n : normals) {
						//spdlog::info(pow2(n.at(0)) + pow2(n.at(1)) + pow2(n.at(2))); //1.0
						colors.push_back((n.as<valf3>() + valf3{ 1.f }) / 2.f);
					}
					data->mesh->SetColors(std::move(colors));

					spdlog::info("Set Normal to Color Success");
				}();
			}


		}
		ImGui::End();
		});
}

void MeshToHEMesh(DenoiseData* data) {
	data->heMesh->Clear();

	if (!data->mesh) {
		spdlog::warn("mesh is nullptr");
		return;
	}

	if (data->mesh->GetSubMeshes().size() != 1) {
		spdlog::warn("number of submeshes isn't 1");
		return;
	}

	std::vector<size_t> indices(data->mesh->GetIndices().begin(), data->mesh->GetIndices().end());
	data->heMesh->Init(indices, 3);

	if (!data->heMesh->IsTriMesh())
		spdlog::warn("HEMesh init fail");

	for (size_t i = 0; i < data->mesh->GetPositions().size(); i++) {
		data->heMesh->Vertices().at(i)->position = data->mesh->GetPositions().at(i);
		data->heMesh->Vertices().at(i)->idx = i;
		//data->heMesh->Vertices().at(i)->bidx = -1;
	}
}

void HEMeshToMesh(DenoiseData* data) {
	if (!data->mesh) {
		spdlog::warn("mesh is nullptr");
		return;
	}

	if (!data->heMesh->IsTriMesh() || data->heMesh->IsEmpty()) {
		spdlog::warn("HEMesh isn't triangle mesh or is empty");
		return;
	}

	data->mesh->SetToEditable();

	const size_t N = data->heMesh->Vertices().size();
	const size_t M = data->heMesh->Polygons().size();
	std::vector<Ubpa::pointf3> positions(N);
	std::vector<Ubpa::pointf2> uvs(N);
	std::vector<uint32_t> indices(M * 3);
	for (size_t i = 0; i < N; i++) {
		positions[i] = data->heMesh->Vertices().at(i)->position;
		uvs[i] = data->heMesh->Vertices().at(i)->uv;
	}

	for (size_t i = 0; i < M; i++) {
		auto tri = data->heMesh->Indices(data->heMesh->Polygons().at(i));
		indices[3 * i + 0] = static_cast<uint32_t>(tri[0]);
		indices[3 * i + 1] = static_cast<uint32_t>(tri[1]);
		indices[3 * i + 2] = static_cast<uint32_t>(tri[2]);
	}
	data->mesh->SetPositions(std::move(positions));
	data->mesh->SetUV(std::move(uvs));
	data->mesh->SetIndices(std::move(indices));
	data->mesh->SetSubMeshCount(1);
	data->mesh->SetSubMesh(0, { 0, M * 3 });
	data->mesh->GenNormals();
	data->mesh->GenTangents();
}


/// <summary>
/// 将平均/高斯曲率归一化，以获得颜色映射（红色表示曲率最大，蓝色最小），参考https://blog.csdn.net/qq_38517015/article/details/105185241
/// </summary>
/// <param name="c">曲率</param>
/// <returns>归一化rgb值</returns>
rgbf ColorMap(float c) {
	float r = 0.8f, g = 1.f, b = 1.f;
	c = c < 0.f ? 0.f : (c > 1.f ? 1.f : c);

	if (c < 1.f / 8.f) {
		r = 0.f;
		g = 0.f;
		b = b * (0.5f + c / (1.f / 8.f) * 0.5f);
	}
	else if (c < 3.f / 8.f) {
		r = 0.f;
		g = g * (c - 1.f / 8.f) / (3.f / 8.f - 1.f / 8.f);
		b = b;
	}
	else if (c < 5.f / 8.f) {
		r = r * (c - 3.f / 8.f) / (5.f / 8.f - 3.f / 8.f);
		g = g;
		b = b - (c - 3.f / 8.f) / (5.f / 8.f - 3.f / 8.f);
	}
	else if (c < 7.f / 8.f) {
		r = r;
		g = g - (c - 5.f / 8.f) / (7.f / 8.f - 5.f / 8.f);
		b = 0.f;
	}
	else {
		r = r - (c - 7.f / 8.f) / (1.f - 7.f / 8.f) * 0.5f;
		g = 0.f;
		b = 0.f;
	}

	return rgbf{ r,g,b };
}
void circle_param(DenoiseData* data, vector<Vertex*>& boundary, Eigen::VectorXf& b_u, Eigen::VectorXf& b_v, float R) {
	float step_angle = PI * 2.0f / static_cast<float>(boundary.size());
	for (size_t i = 0; i < boundary.size(); ++i)
	{
		b_u[boundary[i]->idx] = R * cos(i * step_angle);
		b_v[boundary[i]->idx] = R * sin(i * step_angle);
	}
}
void square_param(DenoiseData* data, vector<Vertex*>& boundary, Eigen::VectorXf& b_u, Eigen::VectorXf& b_v, float length) {
	int side_len = boundary.size() / 4;
	//Fix four corner
	b_u[boundary[0]->idx] = 0.0f;
	b_v[boundary[0]->idx] = 0.0f;

	b_u[boundary[side_len]->idx] = 0.0f;
	b_v[boundary[side_len]->idx] = length;

	b_u[boundary[side_len * 2]->idx] = length;
	b_v[boundary[side_len * 2]->idx] = length;

	b_u[boundary[side_len * 3]->idx] = length;
	b_v[boundary[side_len * 3]->idx] = 0;

	//Fix four boundaries
	float step = length / side_len;

	for (int i = 0; i < side_len; ++i) {
		b_u[boundary[i]->idx] = 0.0f;
		b_v[boundary[i]->idx] = i * step;

	}
	for (int i = side_len + 1, j = 0; i < side_len * 2; ++i, ++j)
	{
		b_u[boundary[i]->idx] = j * step;
		b_v[boundary[i]->idx] = length;

	}

	for (int i = side_len * 2 + 1, j = 0; i < side_len * 3; ++i, ++j)
	{
		b_u[boundary[i]->idx] = length;
		b_v[boundary[i]->idx] = length - j * step;

	}

	for (int i = side_len * 3 + 1, j = 0; i < boundary.size(); ++i, ++j)
	{
		b_u[boundary[i]->idx] = length - j * step;
		b_v[boundary[i]->idx] = 0;

	}
}
void arc_param(DenoiseData* data, vector<Vertex*>& boundary, Eigen::VectorXf& b_u, Eigen::VectorXf& b_v, float R) {
	float length = 0.f;
	//Get closed boundary's step length
	std::vector<float> step;
	for (int i = 0; i < boundary.size() - 1; ++i) {
		float l = norm(boundary[i] - boundary[i + 1]);
		length += l;
		step.push_back(l);
	}
	float l = norm(boundary[boundary.size() - 1] - boundary[0]);
	length += l;
	step.push_back(l);

	float step_angle = 0.0f;
	for (size_t i = 0; i < boundary.size(); ++i)
	{
		b_u[boundary[i]->idx] = R * cos(i * step_angle);
		b_v[boundary[i]->idx] = R * sin(i * step_angle);
		step[i] = 2.0f * PI * step[i] / length;
		step_angle += step[i];
	}
}
float getCotbetween(Vertex* adjV, Vertex* v) {
	auto getCot = []
	(Vertex* a, Vertex* b, Vertex* c)
	{return (a->position - b->position).cos_theta(c->position - b->position); };
	//Calculate alpha and beta
	Vertex* pp;
	Vertex* np;
	auto* adjVE = adjV->HalfEdge();
	while (true) {
		if (v == adjVE->End()) {
			pp = adjVE->Next()->End();
			break;
		}
		else
			adjVE = adjVE->Pair()->Next();
	}
	while (true) {
		if (v == adjVE->Next()->End()) {
			np = adjVE->End();
			break;
		}
		else
			adjVE = adjVE->Pair()->Next();
	}
	float wi = getCot(v, pp, adjV) + getCot(v, np, adjV);
	return wi;
}
void Parameterization(DenoiseData* data) {
	vector<Vertex*>boundary;
	auto& verts = data->heMesh->Vertices();
	int p = -1;
	for (auto* v : verts) {
		if (v->IsOnBoundary()) {
			p = v->idx;
		}
	}
	if (p == -1) {
		spdlog::error("No boundary");
		return;
	}
	while (!verts[p]->visited)
	{
		verts[p]->visited = 1;
		boundary.push_back(verts[p]);
		for (auto* v : verts[p]->AdjVertices()) {
			if (v->IsOnBoundary() && !v->visited) {
				p = v->idx;
				break;
			}
		}

	}
	int n = static_cast<int>(verts.size());
	Eigen::VectorXf b_u(n);
	Eigen::VectorXf b_v(n);
	if (data->type == 0)
		circle_param(data, boundary, b_u, b_v, 1.0f);
	else if (data->type == 1)
		arc_param(data, boundary, b_u, b_v, 1.0f);
	else if (data->type == 2)
		square_param(data, boundary, b_u, b_v, 1.0f);
	Eigen::SparseMatrix<float>A(n, n);
	Eigen::SparseLU<Eigen::SparseMatrix<float>, Eigen::COLAMDOrdering<int> >solver;
	vector<Eigen::Triplet<float>> triplets;

	//Constructing Laplace matrix
	for (auto v : data->heMesh->Vertices()) {
		if (v->IsOnBoundary()) {
			triplets.emplace_back(Eigen::Triplet<float>(v->idx, v->idx, 1.0f));
		}
		else {

			float w = 0;
			for (auto* adjV : v->AdjVertices()) {
				float wi = getCotbetween(adjV, v);
				triplets.emplace_back(Eigen::Triplet<float>(v->idx, adjV->idx, wi));
				w += wi;
			}
			b_u(v->idx) = 0.f;
			b_v(v->idx) = 0.f;
			triplets.emplace_back(Eigen::Triplet<float>(v->idx, v->idx, -w));
		}

	}

	A.setFromTriplets(triplets.begin(), triplets.end());
	solver.analyzePattern(A);
	solver.factorize(A);
	Eigen::VectorXf out_u = solver.solve(b_u);
	Eigen::VectorXf out_v = solver.solve(b_v);
	for (auto v : data->heMesh->Vertices()) {
		int idx = v->idx;
		v->uv = { out_u[idx],out_v[idx] };
		v->position[0] = out_u[idx];
		v->position[1] = out_v[idx];
		v->position[2] = 0.f;

	}


}
void GetMinSurface(DenoiseData* data) {
	int n = static_cast<int>(data->heMesh->Vertices().size());
	Eigen::SparseMatrix<float>A(n, n);
	Eigen::VectorXf b_x(n);
	Eigen::VectorXf b_y(n);
	Eigen::VectorXf b_z(n);
	Eigen::SparseLU<Eigen::SparseMatrix<float>, Eigen::COLAMDOrdering<int> >solver;
	vector<Eigen::Triplet<float>> triplets;

	//Constructing Laplace matrix
	for (auto v : data->heMesh->Vertices()) {
		if (v->IsOnBoundary()) {
			triplets.emplace_back(Eigen::Triplet<float>(v->idx, v->idx, 1.0f));
			b_x(v->idx) = v->position[0];
			b_y(v->idx) = v->position[1];
			b_z(v->idx) = v->position[2];
		}
		else {

			b_x(v->idx) = 0.f;
			b_y(v->idx) = 0.f;
			b_z(v->idx) = 0.f;
			float w = 0;
			for (auto* adjV : v->AdjVertices()) {
				float wi = getCotbetween(adjV, v);
				triplets.emplace_back(Eigen::Triplet<float>(v->idx, adjV->idx, wi));
				w += wi;
			}
			triplets.emplace_back(Eigen::Triplet<float>(v->idx, v->idx, -w));
		}

	}

	A.setFromTriplets(triplets.begin(), triplets.end());
	solver.analyzePattern(A);
	solver.factorize(A);
	Eigen::VectorXf x = solver.solve(b_x);
	Eigen::VectorXf y = solver.solve(b_y);
	Eigen::VectorXf z = solver.solve(b_z);
	for (auto v : data->heMesh->Vertices()) {

		if (!v->IsOnBoundary()) {
			int idx = v->idx;
			v->position[0] = x[idx];
			v->position[1] = y[idx];
			v->position[2] = z[idx];
		}

	}
	spdlog::info("Data transform successful");
}