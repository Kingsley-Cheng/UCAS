//
// Created by Kingsley Cheng on 2023/3/23.
//
#include "tgaimage.h"
#include "model.h"

const TGAColor white = TGAColor(255, 255, 255, 255);
const TGAColor red = TGAColor(255, 0, 0, 255);

void line(int x0, int y0, int x1, int y1, TGAImage &image, TGAColor color) {
    bool steep = false; // 标记当前斜率的绝对值是否大于 1
    if (std::abs(x1 - x0) < std::abs(y1 - y0)) {
        // 若斜率大于 1， 我们将做关于 y=x 对称
        steep = true;
        std::swap(x0, y0);
        std::swap(x1, y1);
    }
    // 判断是 x0 大还是 x1 大
    if (x0 > x1) {
        std::swap(x0, x1);
        std::swap(y0, y1);
    }
    int dx = x1 - x0;
    int dy = y1 - y0;
    int derror2 = 2 * std::abs(dy);
    int error2 = 0;
    int y = y0;
    for (int x = x0; x <= x1; x++) {
        if (steep)
            image.set(y, x, color);
        else
            image.set(x, y, color);
        // Bresenham’s Linear Algorithm
        error2 += derror2;
        if (error2 > dx) {
            error2 -= 2 * dx;
            y += (y1 > y0 ? 1 : -1);
        }
    }
}


int main(int argc, char **argv) {
    // 创建环境
    int width = 500;
    int height = 500;
    TGAImage image(width, height, TGAImage::RGB);
    // 实例化模型
    Model model("../obj/african_head.obj");
    // 访问所有的三角形
    for (int i = 0; i < model.nfaces(); i++) {
        std::vector<int> face = model.face(i); // 取出一个三角形的面
        // 访问其三个顶点
        for (int j = 0; j < 3; j++) {
            // 遍历三角形面片的三个顶点中的两个
            Vec3f v0 = model.vert(face[j]);
            Vec3f v1 = model.vert(face[(j + 1) % 3]);
            // 做视口变换
            int x0 = (v0.x + 1.) * width / 2.;
            int x1 = (v1.x + 1.) * width / 2.;
            int y0 = (v0.y + 1.) * height / 2.;
            int y1 = (v1.y + 1.) * height / 2.;
            // 画三角形其中一边
            line(x0, y0, x1, y1, image, red);
        }
    }
    image.flip_vertically();
    image.write_tga_file("output.tga");
    return 0;
}