#ifndef __RASTERIZER_H__
#define __RASTERIZER_H__

#include"tgaimage.h"
#include"geometry.h"

extern Matrix ModelT;
extern Matrix ViewT;
extern Matrix ProjectionT;
extern Matrix ViewportT;

void viewport_trans(int w, int h);

void projection_trans(float coeff);

void view_trans(Vec3f eye, Vec3f gaze, Vec3f up);

void model_trans(float angle, Vec3f ratio, Vec3f position);

struct Shader {
    virtual ~Shader();

    virtual Vec4f vertex(int iface, int ivert) = 0;

    virtual bool fragment(Vec3f bar, TGAColor &color) = 0;
};

void triangle(Vec4f *pts, Shader &shader, TGAImage &image, TGAImage &zbuffer);

#endif//__RASTERIZER_H__