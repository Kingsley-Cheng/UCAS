#ifndef __MODEL_H__
#define __MODEL_H__

#include <vector>
#include "geometry.h"
#include "tgaimage.h"

class Model {
private:
    // 从上到下顶点
    std::vector<Vec3f> verts_;
    // vertex, uv, normal
    std::vector<std::vector<Vec3i> > faces_;
    // 从上到下顶点法向量
    std::vector<Vec3f> norms_;
    // 从上到下纹理映射坐标
    std::vector<Vec2f> uv_;
    TGAImage diffusemap_;

    void load_texture(std::string filename, const char *suffix, TGAImage &img);

public:
    Model(const char *filename);

    ~Model();

    int nverts();

    int nfaces();

    Vec3f vert(int i);

    // 获取纹理映射坐标
    Vec2i uv(int iface, int nvert);

    // 获取对应纹理坐标颜色
    TGAColor diffuse(Vec2i uv);

    std::vector<int> face(int idx);
};

#endif //__MODEL_H__