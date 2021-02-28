//From 优秀作业参考 http://staff.ustc.edu.cn/~lgliu/Courses/GAMES102_2020/default.html#GAMES

#include "DenoiseSystem.h"
#include "../Components/DenoiseData.h"
#include <_deps/imgui/imgui.h>
#include <spdlog/spdlog.h>
#include <cstring>
#include <cmath>

#define EPSILON 1E-4F //取ZERO<float>有时会报错

using namespace std;
using namespace Ubpa;

void MeshToHEMesh(DenoiseData* data);
void HEMeshToMesh(DenoiseData* data);
float GetTriangleArea(Vertex* v1, Vertex* v2, Vertex* v3);
float GetAmixed(Vertex* v);
valf3 GetMeanCurvatureOperator(Vertex* v);
float GetGaussianCurvature(Vertex* v);
rgbf ColorMap(float c);

void DenoiseSystem::OnUpdate(Ubpa::UECS::Schedule& schedule) {
	schedule.RegisterCommand([](Ubpa::UECS::World* w) {
		auto data = w->entityMngr.GetSingleton<DenoiseData>();
		if (!data)
			return;

		if (ImGui::Begin("Denoise")) {
			ImGui::Text("Operation:"); ImGui::SameLine();
			if (ImGui::Button("Generate min surface")) {
				[&]() {
					data->copy = *data->mesh;
					MeshToHEMesh(data);
					if (!data->heMesh->IsTriMesh() || data->heMesh->IsEmpty()) {
						spdlog::warn("HEMesh isn't triangle mesh or is empty");
						return;
					}

					for (int i = 0; i < data->num_iterations; i++) {
						for (auto* v : data->heMesh->Vertices()) {
							if (!v->IsOnBoundary()) {
								valf3 Hn = GetMeanCurvatureOperator(v) / 2.f;
								v->position -= data->lambda * Hn;
							}
						}
					}
					HEMeshToMesh(data);
					spdlog::info("Generate minimal surface success");
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
			ImGui::SameLine();
			if (ImGui::Button("Mean Curvature")) {
				[&]() {
					if (!data->mesh) {
						spdlog::warn("mesh is nullptr");
						return;
					}

					MeshToHEMesh(data);
					if (!data->heMesh->IsTriMesh() || data->heMesh->IsEmpty()) {
						spdlog::warn("HEMesh isn't triangle mesh or is empty");
						return;
					}

					data->mesh->SetToEditable();
					vector<float> cs;
					for (auto* v : data->heMesh->Vertices())
						cs.push_back(GetMeanCurvatureOperator(v).norm() / 2.f); //根据原论文，平均曲率为算子K长度的1/2
					std::vector<rgbf> colors;
					for (auto c : cs)
						colors.push_back(ColorMap(c));
					data->mesh->SetColors(std::move(colors));

					spdlog::info("Set Mean Curvature to Color Success");
				}();
			}
			ImGui::SameLine();
			if (ImGui::Button("Gaussian Curvature")) {
				[&]() {
					if (!data->mesh) {
						spdlog::warn("mesh is nullptr");
						return;
					}

					MeshToHEMesh(data);
					if (!data->heMesh->IsTriMesh() || data->heMesh->IsEmpty()) {
						spdlog::warn("HEMesh isn't triangle mesh or is empty");
						return;
					}

					data->mesh->SetToEditable();
					vector<float> cs;
					for (auto* v : data->heMesh->Vertices())
						cs.push_back(GetGaussianCurvature(v));
					std::vector<rgbf> colors;
					for (auto c : cs)
						colors.push_back(ColorMap(c));
					data->mesh->SetColors(std::move(colors));

					spdlog::info("Set Gaussian Curvature to Color Success");
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
	std::vector<uint32_t> indices(M * 3);
	for (size_t i = 0; i < N; i++)
		positions[i] = data->heMesh->Vertices().at(i)->position;
	for (size_t i = 0; i < M; i++) {
		auto tri = data->heMesh->Indices(data->heMesh->Polygons().at(i));
		indices[3 * i + 0] = static_cast<uint32_t>(tri[0]);
		indices[3 * i + 1] = static_cast<uint32_t>(tri[1]);
		indices[3 * i + 2] = static_cast<uint32_t>(tri[2]);
	}
	data->mesh->SetPositions(std::move(positions));
	data->mesh->SetIndices(std::move(indices));
	data->mesh->SetSubMeshCount(1);
	data->mesh->SetSubMesh(0, { 0, M * 3 });
	data->mesh->GenNormals();
	data->mesh->GenTangents();
}

float GetTriangleArea(Vertex* v0, Vertex* v1, Vertex* v2) {
	return 0.5f * v0->position.distance(v1->position) * v0->position.distance(v2->position) * (v1->position - v0->position).sin_theta(v2->position - v0->position);
}

float GetAmixed(Vertex* v) {
	float A_mixed = 0.f;
	if (v->IsOnBoundary()) return A_mixed;

	for (auto* adjV : v->AdjVertices()) {
		Vertex* np;
		auto* adjVE = adjV->HalfEdge();
		while (true) {
			if (v == adjVE->Next()->End()) {
				np = adjVE->End();
				break;
			}
			else
				adjVE = adjVE->Pair()->Next();
		}

		if ((adjV->position - v->position).dot(np->position - v->position) >= 0.f && (v->position - adjV->position).dot(np->position - adjV->position) >= 0.f && (v->position - np->position).dot(adjV->position - np->position) >= 0.f) {
			//T is non-obtuse
			if (GetTriangleArea(v, adjV, np) > EPSILON)
				A_mixed += (v->position.distance2(adjV->position) * (v->position - np->position).cot_theta(adjV->position - np->position) + v->position.distance2(np->position) * (v->position - adjV->position).cot_theta(np->position - adjV->position)) / 8.f;
		}
		else {
			if ((adjV->position - v->position).dot(np->position - v->position) < 0.f) {
				//T is obtuse && the angle of T at v is obtuse
				A_mixed += GetTriangleArea(v, adjV, np) / 2.f;
			}
			else {
				//T is obtuse && the angle of T at v is non-obtuse
				A_mixed += GetTriangleArea(v, adjV, np) / 4.f;
			}
		}
	}

	return A_mixed;
}

valf3 GetMeanCurvatureOperator(Vertex* v) {
	valf3 c{ 0.f };
	if (v->IsOnBoundary()) return c;

	float A_mixed = GetAmixed(v);
	if (A_mixed < EPSILON) return c;

	for (auto* adjV : v->AdjVertices()) {
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

		if (GetTriangleArea(v, adjV, pp) > EPSILON && GetTriangleArea(v, adjV, np) > EPSILON) {
			float cot_alpha = (adjV->position - pp->position).cot_theta(v->position - pp->position);
			float cot_beta = (adjV->position - np->position).cot_theta(v->position - np->position);
			c += (cot_alpha + cot_beta) * (v->position - adjV->position);
		}
	}

	return c / (A_mixed * 2.f);
}

float GetGaussianCurvature(Vertex* v) {
	float c{ 0.f };
	if (v->IsOnBoundary()) return c;

	float A_mixed = GetAmixed(v);
	if (A_mixed < EPSILON) return c;

	c = float{ 2.f * PI<float> };
	for (auto* adjV : v->AdjVertices()) {
		Vertex* np;
		auto* adjVE = adjV->HalfEdge();
		while (true) {
			if (v == adjVE->Next()->End()) {
				np = adjVE->End();
				break;
			}
			else
				adjVE = adjVE->Pair()->Next();
		}

		if (GetTriangleArea(v, adjV, np) > EPSILON)
			c -= acos((adjV->position - v->position).cos_theta(np->position - v->position));
	}

	return c / A_mixed;
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