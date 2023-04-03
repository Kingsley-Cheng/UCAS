//
// Created by Kingsley Cheng on 2023/3/23.
//
// includes
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>
#include "tgaimage.h"
#include "model.h"
#include "tgaimage.h"

// 环境高宽
const int width = 800;
const int height = 800;
const float MY_PI = 3.1415926;

// 初始化
// 模型
Model *model = NULL;
// 深度缓冲
int *zbuffer = NULL;
// 光照方向，默认垂直打光
Vec3f light_dir(0,0,-1);


// 将一个三维向量变为齐次向量
Matrix v2m(Vec3f v){
    Matrix m(4,1);
    for(int i =0; i<3;i++)
        m[i][0] = v[i];
    m[3][0] = 1.f;
    return m;
}

// 将一个其次向量转化为三维向量
Vec3f m2v(Matrix m){
    return Vec3f (m[0][0]/m[3][0],m[1][0]/m[3][0], m[2][0]/m[3][0]);
}


// Model trans
Matrix model_trans(float angle,  Vec3f ratio, Vec3f position)
{
    angle = angle* MY_PI/180.f;
    Matrix rotation = Matrix::identity(4);
    Matrix scale = Matrix::identity(4);
    Matrix translate = Matrix::identity(4);

    // rotation
   rotation[0][0] = std::cos(angle);
   rotation[0][2] = std::sin(angle);
   rotation[2][0] = -std::sin(angle);
   rotation[2][2] = std::cos(angle);

   // ratio
   for(int j=0; j<3; j++)
       scale[j][j] = ratio[j];

   // translate
    for (int i = 0; i < 3; ++i)
        translate[i][3] = - position[i];

    return translate * scale * rotation;
}

// Camera trans.
Matrix camera_trans(Vec3f e, Vec3f g, Vec3f t){
    // e: 相机位置
    // g: 观察方向
    // t: 视点正上方向
    Matrix result = Matrix::identity(4);
    // 移动到原点
    Matrix origin = Matrix::identity(4);
    for(int i=0; i<3;i++)
        origin[i][3] = -e[i];

    // 计算相机所在坐标系的三个单位向量
    Vec3f w = g.normalize()*-1;
    Vec3f u = (t^w).normalize();
    Vec3f v = w^u;
    // 转到世界坐标系
    for (int i = 0; i < 3; ++i) {
        result[0][i] = u[i];
        result[1][i] = v[i];
        result[3][i] = w[i];
    }
    return result * origin;
}

// Projection trans,
Matrix projection_trans(float eye_fov, float aspect_ratio, float zNear, float zFar){
    Matrix projection = Matrix::identity(4);
    float alpha = 0.5 * eye_fov * MY_PI / 180.0f;
    float yTop = -zNear * std::tan(alpha);
    float yBottom = -yTop;
    float xRight = yTop * aspect_ratio;
    float xLeft = -xRight;
    Matrix M_persp = Matrix::identity(4);
    Matrix M_ortho = Matrix::identity(4);
    Matrix M_trans = Matrix::identity(4);
    M_persp[0][0] = zNear;
    M_persp[1][1] = zNear;
    M_persp[2][2] = zNear + zFar;
    M_persp[2][3] = -zFar * zNear;
    M_persp[3][2] = 1;
    M_persp[3][3] = 0;
    M_trans[0][3] = -(xLeft + xRight) / 2;
    M_trans[1][3] = -(yTop + yBottom) / 2;
    M_trans[2][3] = -(zNear + zFar) / 2;
    M_ortho[0][0] = 2 / (xRight - xLeft);
    M_ortho[1][1] = 2 / (yTop - yBottom);
    M_ortho[2][2] =  2 / (zNear - zFar);
    M_ortho = M_ortho * M_trans;
    projection = M_ortho * M_persp * projection;
    return projection;
}


// Viewport trans.
Matrix viewport_trans(int w, int h){
    Matrix m = Matrix::identity(4);
    // scale trans.
    m[0][0] = w/2.f;
    m[1][1] = h/2.f;

    //translate trans.
    m[0][3] = w/2.f;
    m[1][3] = h/2.f;
    return m;
}

// 定义两种颜色
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
Vec3f barycentric(Vec3i *pts, Vec3i P) {
    // 重心坐标的权重与之垂直
    Vec3f u = Vec3f(pts[2].x - pts[0].x, pts[1].x - pts[0].x, pts[0].x - P.x) ^
              Vec3f(pts[2].y - pts[0].y, pts[1].y - pts[0].y, pts[0].y - P.y);
    // z = 0 等价于三角形两边平行，退化情况
    if (std::abs(u.z) < 1) return Vec3f(-1, 1, 1);
    // 权重归一化
    return Vec3f(1 - (u.x + u.y) / u.z, u.x / u.z, u.y / u.z);
}

// 输入三角形的三个顶点
void triangle(Vec3i *pts, Vec2i *texture_pts, int *zBuffer, TGAImage &image, float intensity) {
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
    Vec3i P;
    Vec2i uv;
    TGAColor color;
    // 通过一个向量的重心坐标判断是否在三角形内
    for (P.x = bboxmin.x; P.x <= bboxmax.x; P.x++)
        for (P.y = bboxmin.y; P.y <= bboxmax.y; P.y++) {
            // 计算该点的三角形重心坐标
            Vec3f bc_screen = barycentric(pts, P);
            // 判断是否在三角形中
            if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0) continue;
            // 用重心坐标计算像素的深度 z 值
            P.z = bc_screen.x * pts[0].z + bc_screen.y * pts[1].z + bc_screen.z * pts[2].z;
            // 判断是否需要写入深度缓冲
            if (zBuffer[int(P.x + P.y * width)] < P.z) {
                zBuffer[int(P.x + P.y * width)] = P.z;
                // 用重心坐标获取纹理坐标
                uv.x = bc_screen.x * texture_pts[0].x + bc_screen.y *
                                                        texture_pts[1].x + bc_screen.z * texture_pts[2].x;
                uv.y = bc_screen.x * texture_pts[0].y + bc_screen.y *
                                                        texture_pts[1].y + bc_screen.z * texture_pts[2].y;
                // 获取纹理颜色
                color = model->diffuse(uv);
                for (int j = 0; j < 3; j++)
                    color.raw[j] *= intensity;
                image.set(P.x, P.y, color);
            }
        }
}


int main(int argc, char **argv) {
//     实例化模型
    if(2==argc)
        model = new Model(argv[1]);
    else
        model = new Model("../obj/african_head.obj");
    // 初始化深度缓冲区
    zbuffer = new int[width * height];
    for (int i = 0; i < width * height; i++)
        zbuffer[i] = std::numeric_limits<int>::min();
    // 初始化环境
    TGAImage image(width, height, TGAImage::RGB);

    // 照相机方向
    Vec3f e(0,0,4);
    Vec3f g(0,0,-1);
    Vec3f t(0,1,0);

    // Drawing
    Matrix Camera = camera_trans(e,g,t);
    Matrix ViewPort = viewport_trans(width, height);
    Matrix Modeltran =  model_trans(20,Vec3f(1,1,1),Vec3f(0,0,0));
    Matrix Projection = projection_trans(45,1,0.1,50);
    // 遍历模型的三角形面片
    for (int i = 0; i < model->nfaces(); i++) {
        std::vector<int> face = model->face(i);
        // 屏幕坐标系
        Vec3i screen_coords[3];
        // 模型世界坐标系
        Vec3f world_coords[3];
        // 纹理坐标系
        Vec2i texture_coords[3];
        for (int j = 0; j < 3; j++) {
            // 获取三角形顶点的世界坐标
            world_coords[j] = model->vert(face[j]);
            // 纹理位置
            texture_coords[j] = model->uv(i, j);
            screen_coords[j] = m2v(ViewPort*Projection*Camera*Modeltran*v2m(world_coords[j]));
            std::cout<<screen_coords[j]<<std::endl;
        }
        // 计算三角形法线方向
        Vec3f n = (world_coords[2] - world_coords[0]) ^ (world_coords[1] - world_coords[0]);
        // 法向量单位化
        n.normalize();
        // 定义光照强度 法向量与光线的内积
        float intensity = n * light_dir;
        // 要求光照强度为正，否则表示在光照背面
        if (intensity > 0) {
            triangle(screen_coords, texture_coords, zbuffer, image, intensity);
        }
    }
    // 垂直翻转，把原点移到左下角
    image.flip_vertically();
    image.write_tga_file("output.tga");
    delete model;
    delete [] zbuffer;
    return 0;
}