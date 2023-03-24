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

// 计算 P 在三角形中的重心坐标
Vec3f barycentric(Vec2i *pts, Vec2i P) {
    // 重心坐标的权重与之垂直
    Vec3f u = Vec3f(pts[2].x - pts[0].x, pts[1].x - pts[0].x, pts[0].x - P.x) ^
              Vec3f(pts[2].y - pts[0].y, pts[1].y - pts[0].y, pts[0].y - P.y);
    // z = 0 等价于三角形两边平行，退化情况
    if (std::abs(u.z) < 1) return Vec3f(-1, 1, 1);
    // 权重归一化
    return Vec3f(1 - (u.x + u.y) / u.z, u.x / u.z, u.y / u.z);
}

// 输入三角形的三个顶点
void triangle(Vec2i *pts, TGAImage &image, TGAColor color) {
    // 寻找三角形的包围盒
    Vec2i bboxmin(image.get_width() - 1, image.get_height() - 1);
    Vec2i bboxmax(0, 0);
    Vec2i clamp(image.get_width() - 1, image.get_height() - 1);
    for (int i = 0; i < 3; i++) {
        // 寻找三角形三个顶点关于x, y 的最大最小值
        bboxmax.x = std::min(clamp.x, std::max(bboxmax.x, pts[i].x));
        bboxmax.y = std::min(clamp.y, std::max(bboxmax.y, pts[i].y));
        bboxmin.x = std::max(0, std::min(bboxmin.x, pts[i].x));
        bboxmin.y = std::max(0, std::min(bboxmin.y, pts[i].y));
    }
    // 通过一个向量的重心坐标判断是否在三角形内
    Vec2i P;
    for (P.x = bboxmin.x; P.x <= bboxmax.x; P.x++)
        for (P.y = bboxmin.y; P.y <= bboxmax.y; P.y++) {
            // 计算该点的三角形重心坐标
            Vec3f bc_screen = barycentric(pts, P);
            if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0) continue;
            image.set(P.x, P.y, color);
        }
}


int main(int argc, char **argv) {
    // 创建环境
    int width = 500;
    int height = 500;
    // 假设光垂直于屏幕
    Vec3f light_dir(0, 0, -1);
    light_dir.normalize();
    TGAImage image(width, height, TGAImage::RGB);
    // 实例化模型
    Model model("../obj/african_head.obj");
    // 遍历模型的三角形面片
    for (int i = 0; i < model.nfaces(); i++) {
        std::vector<int> face = model.face(i);
        // 屏幕坐标系
        Vec2i screen_coords[3];
        // 模型世界坐标系
        Vec3f world_coords[3];
        for (int j = 0; j < 3; j++) {
            Vec3f v = model.vert(face[j]);
            // 做 viewport transform
            screen_coords[j] = Vec2i((v.x + 1.) * width / 2.,
                                     (v.y + 1.) * height / 2.);
            world_coords[j] = v;
        }
        // 计算三角形法线方向
        Vec3f n = (world_coords[2] - world_coords[0]) ^ (world_coords[1] - world_coords[0]);
        // 法向量单位化
        n.normalize();

        // 定义光照强度 法向量与光线的内积
        float intensity = n * light_dir;
        // 要求光照强度为正，否则表示在光照背面
        if (intensity > 0) {
            triangle(screen_coords, image,
                     TGAColor(intensity * 255, intensity * 255, intensity * 255, 255));
        }
    }
    // 垂直翻转，把原点移到左下角
    image.flip_vertically();
    image.write_tga_file("output.tga");
    return 0;
}