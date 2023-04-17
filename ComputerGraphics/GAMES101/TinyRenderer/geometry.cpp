#include "geometry.h"

template<>
template<>
vec<3, int>::
// float 向量转为 int 进行四舍五入
vec(const vec<3, float> &v):
        x(int(v.x + .5f)), y(int(v.y + .5f)), z(int(v.z + .5f)) {}

template<>
template<>
vec<3, float>::
// int 向量转为 float 向量
vec(const vec<3, int> &v):
        x(v.x), y(v.y), z(v.z) {}

template<>
template<>
vec<2, int>::
// float 向量转为 int 进行四舍五入
vec(const vec<2, float> &v):
        x(int(v.x + .5f)), y(int(v.y + .5f)) {}

template<>
template<>
vec<2, float>::
// int 向量转为 float 向量
vec(const vec<2, int> &v):
        x(v.x), y(v.y) {}

