#include <amp.h>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;
using namespace Concurrency;

static const int Width = 640;
static const int Height = 480;
static const int TileWidth = 16;
static const int TileHeight = 16;

template<typename T> T Square(T x) restrict(direct3d, cpu) { return x * x; }
template<typename T> T Clamp(T x, T a, T b) restrict(direct3d, cpu) { return x < a ? a : (x < b ? x : b); }
template<typename T> T Saturate(T x) restrict(direct3d, cpu) { return Clamp(x, T(0), T(1)); }

struct Vector3
{
	explicit Vector3(float xyz) restrict(direct3d, cpu) : x(xyz), y(xyz), z(xyz) {}
	Vector3(float x_, float y_, float z_) restrict(direct3d, cpu) : x(x_), y(y_), z(z_) {}

	Vector3& operator+=(const Vector3& rhs) restrict(direct3d, cpu) { x += rhs.x; y += rhs.y; z += rhs.z; return *this; }
	Vector3& operator-=(const Vector3& rhs) restrict(direct3d, cpu) { x -= rhs.x; y -= rhs.y; z -= rhs.z; return *this; }
	Vector3& operator*=(float rhs) restrict(direct3d, cpu)  { x *= rhs; y += rhs; z *= rhs; return *this; }

	float x, y, z;
};

inline Vector3 operator+(Vector3 lhs, const Vector3& rhs) restrict(direct3d, cpu) { lhs += rhs; return lhs; }
inline Vector3 operator-(Vector3 lhs, const Vector3& rhs) restrict(direct3d, cpu) { lhs -= rhs; return lhs; }
inline Vector3 operator*(Vector3 lhs, float rhs) restrict(direct3d, cpu) { lhs *= rhs; return lhs; }
inline Vector3 operator*(float lhs, Vector3 rhs) restrict(direct3d, cpu) { rhs *= lhs; return rhs; }

inline float Dot(const Vector3& a, const Vector3& b) restrict(direct3d, cpu) { return a.x * b.x + a.y * b.y + a.z * b.z; }

class Sphere
{
public:
	Sphere(const Vector3& center, float radius) restrict(direct3d, cpu) : m_center(center), m_radius(radius) {}

	const Vector3& GetCenter() const restrict(direct3d, cpu) { return m_center; }
	float GetRadius() const restrict(direct3d, cpu) { return m_radius; }

private:
	Vector3 m_center;
	float m_radius;
};

float RayIntersectSphere(const Vector3& rayStart, const Vector3& rayDirection, const Sphere& sphere) restrict(direct3d, cpu)
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

uint32_t MakeRGB(int r, int g, int b) restrict(direct3d, cpu)
{
	return ((r & 0xff) << 24) | ((g & 0xff) << 16) | ((b & 0xff) << 8);
}

uint32_t MakeRGB(float r, float g, float b) restrict(direct3d, cpu)
{
	return MakeRGB(int(Saturate(r) * 255.f), int(Saturate(g) * 255.f), int(Saturate(b) * 255.f));
}

uint32_t MakeRGB(Vector3 v) restrict(direct3d, cpu)
{
	return MakeRGB(v.x, v.y, v.z);
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

inline uint32_t TraceRay(const Vector3& rayStart, const Vector3& rayDirection) restrict(direct3d, cpu)
{
	Sphere s(Vector3(0.f), 5.f);
	if (RayIntersectSphere(rayStart, rayDirection, s) == FLT_MAX)
	{
		return MakeRGB(16, 16, 64);
	}
	else
	{
		return MakeRGB(192, 16, 16);
	}
}

inline uint32_t RaytraceSphereflake(int x, int y) restrict (direct3d)
{
	const auto viewWidth = 15.0f;
	const auto viewHeight = viewWidth * float(Height) / float(Width);
	const Vector3 rayStart(viewWidth * ((float(x) / float(Width)) - 0.5f), 
		viewHeight * ((float(Height - y) / float(Height)) - 0.5f), -10.0f);
	const Vector3 rayDirection(0.f, 0.f, 1.f);
	return TraceRay(rayStart, rayDirection);
}

vector<uint32_t> RaytraceSphereflake()
{
	vector<uint32_t> imageBuffer(Width * Height);
	array_view<uint32_t, 2> imageView(Height, Width, &imageBuffer[0]);

	parallel_for_each(
		imageView.grid.tile<TileWidth, TileHeight>(),
		[=](tiled_index<TileWidth, TileHeight> idx) mutable restrict(direct3d)
		{
			imageView[idx.global] = RaytraceSphereflake(idx.get_x(), idx.get_y());
		}
	);

	return imageBuffer;
}

int main()
{
	auto res = RaytraceSphereflake();
	WritePPM("amp-sphereflake.ppm", res, Width, Height);

	return 0;
}
