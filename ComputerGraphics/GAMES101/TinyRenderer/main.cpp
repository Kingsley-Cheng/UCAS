#include <vector>
#include <iostream>

#include "tgaimage.h"
#include "model.h"
#include "geometry.h"
#include "rasterizer.h"

Model *model = NULL;
const int width = 800;
const int height = 800;

Vec3f light_dir(0.4, 1, 1);
Vec3f eye(0, 0, 0);
Vec3f gaze(1, 1, -4);
Vec3f up(0, 1, 0);

float angle = 0;
Vec3f ratio(1, 1, 1);
Vec3f position(0, 0, 0);

struct GouraudShader : public Shader {
    Vec3f varying_intensity;
    mat<2,3,float> varying_uv;

    virtual Vec4f vertex(int iface, int ivert) {
        varying_uv.set_col(ivert, model->uv(iface, ivert));
        Vec4f Vertex = embed<4>(model->vert(iface, ivert));
        Vertex = ViewportT *ProjectionT* ViewT * ModelT * Vertex;
        varying_intensity[ivert] = std::max(0.f, model->normal(iface, ivert) * light_dir);
        return Vertex;
    }

    virtual bool fragment(Vec3f bar, TGAColor &color) {
        float intensity = varying_intensity * bar;
        Vec2f uv = varying_uv * bar;
        color = model->diffuse(uv) * intensity;
        return false;
    }
};

int main() {
    model = new Model("../obj/african_head.obj");
    //MVP Trans
    model_trans(angle, ratio, position);
    view_trans(eye, gaze, up);
    projection_trans(-1.f/gaze.norm());

    //ViewPort Trans
    viewport_trans(width, height);
    light_dir.normalize();

    TGAImage image(width, height, TGAImage::RGB);
    TGAImage zbuffer(width, height, TGAImage::GRAYSCALE);

    GouraudShader shader;
    for (int i = 0; i < model->nfaces(); i++) {
        Vec4f screen_coords[3];
        for (int j = 0; j < 3; j++) {
            screen_coords[j] = shader.vertex(i, j);
        }
        triangle(screen_coords, shader, image, zbuffer);
    }

    image.flip_vertically();
    zbuffer.flip_vertically();
    image.write_tga_file("output.tga");
    zbuffer.write_tga_file("zbuffer.tga");

    delete model;
    return 0;
}