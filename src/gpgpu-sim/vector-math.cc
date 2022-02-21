#include "vector-math.h"

float3 make_float3(float a, float b, float c)
{
    return {a, b, c};
}
float4 make_float4(float a, float b, float c, float d)
{
    return {a, b, c, d};
}

// Subtraction
float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// Addition
float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

// Multiply elements
float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

float3 operator*(float3 a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

float3 operator*(float s, float3 a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

// Cross Product
float3 cross(float3 a, float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

// Dot Product
float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/* FLOAT 4 */
float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

float clamp(float x, float lo, float hi)
{
    return std::max(std::min(x, hi), lo);
}

float3 min(float3 a, float3 b)
{
    return make_float3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

float3 max(float3 a, float3 b)
{
    return make_float3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

