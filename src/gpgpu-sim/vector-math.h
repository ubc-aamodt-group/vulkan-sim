#ifndef VECTOR_MATH_INCLUDED
#define VECTOR_MATH_INCLUDED
#include "../abstract_hardware_model.h"

float3 make_float3(float a, float b, float c);
float4 make_float4(float a, float b, float c, float d);

float3 operator-(float3 a, float3 b);
float3 operator+(float3 a, float3 b);
float3 operator*(float3 a, float3 b);
float3 operator*(float3 a, float s);
float3 operator*(float s, float3 a);

float3 cross(float3 a, float3 b);
float dot(float3 a, float3 b);

float4 operator-(float4 a, float4 b);
float4 operator+(float4 a, float4 b);
float4 operator*(float4 a, float4 b);

float clamp(float x, float lo, float hi);
float3 min(float3 a, float3 b);
float3 max(float3 a, float3 b);

const float PI = 3.14159265358979323846;

#endif
