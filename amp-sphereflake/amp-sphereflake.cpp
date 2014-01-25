#include <amp.h>
#include <amp_math.h>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;
using namespace concurrency;
using namespace concurrency::fast_math;

static const int Width = 640;
static const int Height = 480;
static const int TileWidth = 16;
static const int TileHeight = 16;

template<typename T> T Square(T x) restrict(amp, cpu) { return x * x; }
template<typename T> T Clamp(T x, T a, T b) restrict(amp, cpu) { return x < a ? a : (x < b ? x : b); }
template<typename T> T Saturate(T x) restrict(amp, cpu) { return Clamp(x, T(0), T(1)); }

const float PI = 3.141592654f;

inline float DegToRad(const float deg) {
	return deg * ((2.0f * PI) / 360.0f);
}

struct Vec3 {
	float x, y, z;
};

inline Vec3 MakeVec3(float x, float y, float z) restrict(amp, cpu) {
	Vec3 v;
	v.x = x; v.y = y; v.z = z;
	return v;
}

struct Vec4 {
	float x, y, z, w;
};

inline Vec4 MakeVec4(float x, float y, float z, float w) restrict(amp, cpu) {
	Vec4 v;
	v.x = x; v.y = y; v.z = z; v.w = w;
	return v;
}

class Vector3
{
public:
	Vector3(const Vec3& v_) restrict(amp, cpu) : v(v_) {}
	explicit Vector3(float xyz) restrict(amp, cpu) : v(MakeVec3(xyz, xyz, xyz)) {}
	Vector3(float x, float y, float z) restrict(amp, cpu) : v(MakeVec3(x, y, z)) {}

	float x() const restrict(amp, cpu) { return v.x; }
	float y() const restrict(amp, cpu) { return v.y; }
	float z() const restrict(amp, cpu) { return v.z; }

	operator Vec3() const restrict(amp, cpu) { return v; }

	Vector3& operator+=(const Vector3& rhs) restrict(amp, cpu) { v.x += rhs.x(); v.y += rhs.y(); v.z += rhs.z(); return *this; }
	Vector3& operator-=(const Vector3& rhs) restrict(amp, cpu) { v.x -= rhs.x(); v.y -= rhs.y(); v.z -= rhs.z(); return *this; }
	Vector3& operator*=(float rhs) restrict(amp, cpu)  { v.x *= rhs; v.y *= rhs; v.z *= rhs; return *this; }

private:
	Vec3 v;
};

inline Vector3 operator+(Vector3 lhs, const Vector3& rhs) restrict(amp, cpu) { lhs += rhs; return lhs; }
inline Vector3 operator-(Vector3 lhs, const Vector3& rhs) restrict(amp, cpu) { lhs -= rhs; return lhs; }
inline Vector3 operator*(Vector3 lhs, float rhs) restrict(amp, cpu) { lhs *= rhs; return lhs; }
inline Vector3 operator*(float lhs, Vector3 rhs) restrict(amp, cpu) { rhs *= lhs; return rhs; }

inline float Dot(const Vector3& a, const Vector3& b) restrict(amp, cpu) { return a.x() * b.x() + a.y() * b.y() + a.z() * b.z(); }
inline Vector3 Normalize(const Vector3& v) restrict(amp) { return v * rsqrt(Dot(v, v)); }
inline Vector3 Normalize(const Vector3& v) restrict(cpu) { return v * (1.0f / sqrt(Dot(v, v))); }

struct Rotor
{
	Rotor(const Vec4& v_) restrict(amp, cpu) : v(v_) {}
	Rotor(float x, float y, float z, float w) restrict(amp, cpu) : v(MakeVec4(x, y, z, w)) {}

	Rotor(const Vector3& axis, float angle) restrict(amp, cpu) {
		const auto halfAngle = 0.5f * angle;
		const auto xyz = Vector3(axis.x(), axis.y(), axis.z()) * -sin(halfAngle);
		v = MakeVec4(xyz.x(), xyz.y(), xyz.z(), cos(halfAngle));
	}

	float x() const restrict(amp, cpu) { return v.x; }
	float y() const restrict(amp, cpu) { return v.y; }
	float z() const restrict(amp, cpu) { return v.z; }
	float w() const restrict(amp, cpu) { return v.w; }

	operator Vec4() const restrict(amp, cpu) { return v; }

	Rotor& operator*=(const Rotor& rhs) restrict(amp, cpu) {
		*this = Rotor(w() * rhs.x() + rhs.w() * x() - y() * rhs.z() + z() * rhs.y(),
					  w() * rhs.y() + rhs.w() * y() + x() * rhs.z() - z() * rhs.x(),
					  w() * rhs.z() + rhs.w() * z() - x() * rhs.y() + y() * rhs.x(),
					  w() * rhs.w() - x() * rhs.x() - y() * rhs.y() - z() * rhs.z());
		return *this;
	}

private:
	Vec4 v;
};

inline Rotor operator*(Rotor lhs, const Rotor& rhs) restrict(amp, cpu) {
	lhs *= rhs;
	return lhs;
}

inline Vector3 Rotate(const Vector3& v, const Rotor& r) restrict(amp, cpu) {
	const auto x2 = Square(r.x());
	const auto y2 = Square(r.y());
	const auto z2 = Square(r.z());
	const auto w2 = Square(r.w());
	const auto wx = r.w() * r.x();
	const auto wy = r.w() * r.y();
	const auto wz = r.w() * r.z();
	const auto xy = r.x() * r.y();
	const auto xz = r.x() * r.z();
	const auto yz = r.y() * r.z();

	return Vector3 ((1.f - 2.f * (y2 + z2)) * v.x() + 2.f * (xy - wz) * v.y() + 2.f * (xz + wy) * v.z(),
					2.f * (xy + wz) * v.x() + (1.f - 2.f * (x2 + z2)) * v.y() + 2.f * (yz - wx) * v.z(),
					2.f * (xz - wy) * v.x() + 2.f * (yz + wx) * v.y() + (1.f - 2.f * (x2 + y2)) * v.z());
}

class Sphere
{
public:
	Sphere(const Vector3& center_, float radius_) restrict(amp, cpu) : center(center_), radius(radius_) {}

	Vector3 GetCenter() const restrict(amp, cpu) { return center; }
	float GetRadius() const restrict(amp, cpu) { return radius; }

private:
	Vec3 center;
	float radius;
};

float RayIntersectSphere(const Vector3& rayStart, const Vector3& rayDirection, const Sphere& sphere) restrict(amp, cpu)
{
	const auto v = rayStart - sphere.GetCenter();
	const auto a = Dot(rayDirection, rayDirection);
	const auto minusB = -2.f * Dot(rayDirection, v);
	const auto c = Dot(v, v) - Square(sphere.GetRadius());
	const auto discrim = (Square(minusB) - 4.f * a * c);
	if (discrim < 0.f)
		return FLT_MAX;

	const auto sqrtDiscrim = sqrt(discrim);
	const auto tMax = minusB + sqrtDiscrim;
	if (tMax < 0.f)
		return FLT_MAX;

	const auto denominator = 1.f / (2.f * a);
	const auto tMin = minusB - sqrtDiscrim;
	return (tMin < 0.f ? tMax : tMin) * denominator;
}

class SphereFlake {
public:
	SphereFlake() 
		: radius(5.0f), 
		  scaleFactor(1.0f / 3.0f), 
		  topChildrenZRotation(55.0f), 
		  bottomChildrenZRotation(110.0f)
	{
		int i = 0;
		const Rotor topChildrenZRotor(Vector3(0, 0, 1), DegToRad(topChildrenZRotation));
		const Rotor bottomChildrenZRotor(Vector3(0, 0, 1), DegToRad(bottomChildrenZRotation));

		// The first 3 entries are for the "top" children
		for (int i = 0; i < 3; ++i) {
			childTransforms[i] = topChildrenZRotor * Rotor(Vector3(0, 1, 0), DegToRad(-30.0f-120.0f*i));
		}

		// The last 4 entries are for the "bottom" children
		for (int i = 0; i < 4; ++i) {
			childTransforms[i + 3] = bottomChildrenZRotor * Rotor(Vector3(0, 1, 0), DegToRad(-45.0f -90.0f*i));
		}

		levelColors[0] = MakeVec3(1.0f, 0.1f, 0.1f);
		levelColors[1] = MakeVec3(0.1f, 1.0f, 0.1f);
		levelColors[2] = MakeVec3(0.1f, 0.1f, 1.0f);
		levelColors[3] = MakeVec3(1.0f, 1.0f, 0.1f);
		levelColors[4] = MakeVec3(0.1f, 1.0f, 1.0f);
	}

	float RayIntersect(const Vector3& rayStart, const Vector3& rayDirection, Vector3& hitCenter, int& hitLevel) const restrict(amp, cpu);

	Vector3 GetLevelColor(int level) const restrict(amp, cpu) { return levelColors[level % MaxDepth]; }

private:
	static const int MaxDepth = 5;
	static const int NumChildren = 7;
	const float radius; // Radius of initial sphere
	const float scaleFactor;	// Controls how much smaller each child sphere flake is.
	const float topChildrenZRotation;
	const float bottomChildrenZRotation;

	Vec4 childTransforms[NumChildren];
	Vec3 levelColors[MaxDepth];
};

class ChildStack {
public:
	struct ChildInfo {
		Vec3 sphereCenter;
		float sphereRadius;
		Vec4 rotation;
		int level;
	};

	ChildStack() restrict(amp, cpu) : top(0) {}

	void Push(const ChildInfo& childInfo) restrict(amp, cpu) { stack[top++] = childInfo; }
	ChildInfo Pop() restrict(amp, cpu) { return stack[--top]; }
	bool Empty() const restrict(amp, cpu) { return top == 0; }
	bool Full() const restrict(amp, cpu) { return top == StackSize; }

private:
	static const int StackSize = 64;
	ChildInfo stack[StackSize];
	int top;
};

float SphereFlake::RayIntersect(const Vector3& rayStart, const Vector3& rayDirection, Vector3& hitCenter, int& hitLevel) const restrict(amp, cpu) {
	ChildStack::ChildInfo ci;
	ci.sphereCenter = Vector3(0.0f);
	ci.sphereRadius = radius;
	ci.rotation = Rotor(Vector3(0.0f, 0.0f, 1.0f), 0.0f);
	ci.level = 0;
	ChildStack cs;
	cs.Push(ci);
	float res = FLT_MAX;
	while (!cs.Empty()) {
		ci = cs.Pop();
		const Sphere boundSphere(ci.sphereCenter, ci.sphereRadius * 2.0f);
		const auto boundsTest = RayIntersectSphere(rayStart, rayDirection, boundSphere);
		if (boundsTest >= res)
			continue;
		const auto hitTest = RayIntersectSphere(rayStart, rayDirection, Sphere(ci.sphereCenter, ci.sphereRadius));
		if (hitTest < res) {
			res = hitTest;
			hitCenter = ci.sphereCenter;
			hitLevel = ci.level;
		}
		if (ci.level > MaxDepth)
			continue;
		for (int i = 0; i < NumChildren; ++i) {
			if (cs.Full())
				break;
			Rotor childRotation(Rotor(childTransforms[i]) * Rotor(ci.rotation));
			Sphere childSphere(ci.sphereCenter + Rotate(Vector3(0.0f, (1.0f + scaleFactor) * ci.sphereRadius, 0.0f), childRotation),
				ci.sphereRadius * scaleFactor);
			ChildStack::ChildInfo newCi;
			newCi.sphereCenter = childSphere.GetCenter();
			newCi.sphereRadius = childSphere.GetRadius();
			newCi.rotation = childRotation;
			newCi.level = ci.level + 1;
			cs.Push(newCi);
		}
	}
	return res;
}

uint32_t MakeRGB(int r, int g, int b) restrict(amp, cpu)
{
	return ((r & 0xff) << 24) | ((g & 0xff) << 16) | ((b & 0xff) << 8);
}

uint32_t MakeRGB(float r, float g, float b) restrict(amp, cpu)
{
	return MakeRGB(int(Saturate(r) * 255.f), int(Saturate(g) * 255.f), int(Saturate(b) * 255.f));
}

uint32_t MakeRGB(Vector3 v) restrict(amp, cpu)
{
	return MakeRGB(v.x(), v.y(), v.z());
}

class Color
{
public:
	explicit Color(uint32_t rgba) : m_rgba(rgba) {}
	Color(int r, int g, int b) : m_rgba(MakeRGB(r, g, b)) {}
	Color(float r, float g, float b) : m_rgba(MakeRGB(r * 255.f, g * 255.f, b * 255.f)) {}

	int GetR() { return (m_rgba & (0xff << 24)) >> 24; }
	int GetG() { return (m_rgba & (0xff << 16)) >> 16; }
	int GetB() { return (m_rgba & (0xff <<  8)) >>  8; }
	int GetA() { return (m_rgba & (0xff <<  0)) >>  0; }

private:
	uint32_t m_rgba;
};

void WritePPM(const char* filename, const vector<uint32_t>& imageData, int imageWidth, int imageHeight)
{
	fstream ppmFileStream(filename, ios_base::out | ios_base::trunc);

	ppmFileStream << "P3" << endl << imageWidth << endl << imageHeight << endl << 255 << endl;
	for (int y = 0; y < imageHeight; ++y)
	{
		for (int x = 0; x < imageWidth; ++x)
		{
			Color color(imageData[y * imageWidth + x]);
			ppmFileStream << color.GetR() << " " << color.GetG() << " " << color.GetB() << " ";
		}
		ppmFileStream << endl;
	}
}

inline uint32_t TraceRay(const Vector3& rayStart, const Vector3& rayDirection, const SphereFlake& sphereFlake) restrict(amp, cpu)
{
	Sphere s(Vector3(0.f), 5.f);
	Vector3 hitCenter(0.0f);
	int hitLevel = 0;
	const auto t = sphereFlake.RayIntersect(rayStart, rayDirection, hitCenter, hitLevel);
	if (t == FLT_MAX)
	{
		return MakeRGB(16, 16, 64);
	}
	else
	{
		const Vector3 hitPos = rayStart + t * rayDirection;
		Vector3 hitNormal = Normalize(hitPos - hitCenter);
		const Vector3 lightPos(100.0f, 100.0f, -100.0f);
		const auto L = Normalize(lightPos - hitPos);
		return MakeRGB(sphereFlake.GetLevelColor(hitLevel) * Dot(hitNormal, L));
	}
}

inline uint32_t RaytraceSphereflake(int x, int y, const SphereFlake& sphereFlake) restrict (amp, cpu)
{
	const auto viewWidth = 20.0f;
	const auto viewHeight = viewWidth * float(Height) / float(Width);
	const Vector3 rayStart(viewWidth * ((float(x) / float(Width)) - 0.5f), 
		viewHeight * ((float(Height - y) / float(Height)) - 0.5f), -10.0f);
	const Vector3 rayDirection(0.f, 0.f, 1.f);
	return TraceRay(rayStart, rayDirection, sphereFlake);
}

vector<uint32_t> RaytraceSphereflakeGPU()
{
	vector<uint32_t> imageBuffer(Width * Height);
	array_view<uint32_t, 2> imageView(Height, Width, &imageBuffer[0]);
	SphereFlake sphereFlake;

	parallel_for_each(
		imageView.extent.tile<TileWidth, TileHeight>(),
		[=](tiled_index<TileWidth, TileHeight> idx) restrict(amp)
		{
			imageView[idx.global] = RaytraceSphereflake(idx.global[0], idx.global[1], sphereFlake);
		}
	);

	return imageBuffer;
}

vector<uint32_t> RaytraceSphereflakeCPU()
{
	vector<uint32_t> imageBuffer(Width * Height);
	SphereFlake sphereFlake;

	for (int y = 0; y < Height; ++y) {
		for (int x = 0; x < Width; ++x) {
			imageBuffer[y * Width + x] = RaytraceSphereflake(x, y, sphereFlake);
		}
	}

	return imageBuffer;
}

int main()
{
	auto res = RaytraceSphereflakeGPU();
	WritePPM("amp-sphereflake.ppm", res, Width, Height);

	auto resCPU = RaytraceSphereflakeCPU();
	WritePPM("cpu-sphereflake.ppm", resCPU, Width, Height);

	return 0;
}
