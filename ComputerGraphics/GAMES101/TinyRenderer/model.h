#ifndef __MODEL_H
#define __MODEL_H

#include <vector>
#include <string>
#include "geometry.h"
#include "tgaimage.h"

class Model {
private:
    // 顶点
    std::vector<Vec3f> verts_;
    // 面片(v/uv/yn)
    std::vector<std::vector<Vec3i> > faces_;
    // 法向量
    std::vector<Vec3f> norms_;
    // 纹理坐标
    std::vector<Vec2f> uv_;
    // 纹理图
    TGAImage diffusemap_;
    TGAImage normalmap_;
    TGAImage specularmap_;

    void load_texture(std::string filename, const char *suffix, TGAImage &img);

public:
    Model(const char *filename);

    ~Model();

    // 顶点数量
    int nverts();

    // 面片数量
    int nfaces();

    // 获取指定三角形面片特定顶点的法向量
    Vec3f normal(int iface, int ivert);

    // 纹理坐标的法向量
    Vec3f normal(Vec2f uv);

    // 返回顶点
    Vec3f vert(int i);

    // 返回指定三角形面片特定顶点坐标
    Vec3f vert(int iface, int ivert);

    // 返回指定三角形面片特定顶点纹理坐标
    Vec2f uv(int iface, int ivert);

    // 对应纹理坐标返回纹理图颜色
    TGAColor diffuse(Vec2f uv);

    float specular(Vec2f uv);

    // 返回面片
    std::vector<int> face(int idx);
};

#endif //__MODEL_H