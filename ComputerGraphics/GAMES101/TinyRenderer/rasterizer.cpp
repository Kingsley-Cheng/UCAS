#include<cmath>
#include<limits>
#include<cstdlib>
#include "rasterizer.h"

const float MY_PI = 3.1415926;
Matrix ModelT;
Matrix ViewT;
Matrix ProjectionT;
Matrix ViewportT;

Shader::~Shader() {}

void viewport_trans(int w, int h) {

    ViewportT = Matrix::identity();
    // scale trans.
    ViewportT[0][0] = w / 2.f;
    ViewportT[1][1] = h / 2.f;
    ViewportT[2][2] = 255.f / 2.f;

    //translate trans.
    ViewportT[0][3] = w / 2.f;
    ViewportT[1][3] = h / 2.f;
    ViewportT[2][3] = 255.f / 2.f;
}

void projection_trans(float coeff) {
    ProjectionT = Matrix::identity();
    ProjectionT[3][2] = coeff;
}

void view_trans(Vec3f eye, Vec3f gaze, Vec3f up) {
    // e: 相机位置
    // g: 观测方向，从模型到相机位置
    // t: 视点正上方向
    Matrix result = Matrix::identity();
    // 移动到原点
    Matrix origin = Matrix::identity();
    for (int i = 0; i < 3; i++)
        origin[i][3] = -eye[i];

    // 计算相机所在坐标系的三个单位向量
    Vec3f w = (gaze*-1).normalize();
    Vec3f u = cross(up, w).normalize();
    Vec3f v = cross(w, u);
    // 转到世界坐标系
    for (int i = 0; i < 3; i++) {
        result[0][i] = u[i];
        result[1][i] = v[i];
        result[2][i] = w[i];
    }
    ViewT = result * origin;
}

void model_trans(float angle, Vec3f ratio, Vec3f position) {
    angle = angle * MY_PI / 180.f;
    Matrix rotation = Matrix::identity();
    Matrix scale = Matrix::identity();
    Matrix translate = Matrix::identity();

    // rotation
    rotation[0][0] = std::cos(angle);
    rotation[0][2] = std::sin(angle);
    rotation[2][0] = -std::sin(angle);
    rotation[2][2] = std::cos(angle);

    // ratio
    for (int j = 0; j < 3; j++)
        scale[j][j] = ratio[j];

    // translate
    for (int i = 0; i < 3; ++i)
        translate[i][3] = -position[i];

    ModelT = translate * scale * rotation;
}

Vec3f barycentric(Vec2f A, Vec2f B, Vec2f C, Vec2f P) {
    Vec3f s[2];
    for (int i = 2; i--;) {
        s[i][0] = C[i] - A[i];
        s[i][1] = B[i] - A[i];
        s[i][2] = A[i] - P[i];
    }

    // 重心坐标的权重与之垂直
    Vec3f u = cross(s[0], s[1]);

    if (std::abs(u[2]) > 1e-2)
        // 权重归一化
        return Vec3f(1.f - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);
    // z = 0 等价于三角形两边平行，退化情况
    return Vec3f(-1, 1, 1);
}

void triangle(Vec4f *pts, Shader &shader, TGAImage &image, TGAImage &zbuffer) {
    Vec2f bboxmin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    Vec2f bboxmax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
    for (int i = 0; i < 3; i++)
        // 寻找三角形三个顶点关于x, y 的最大最小值
        for (int j = 0; j < 2; j++) {
            bboxmin[j] = std::min(bboxmin[j], pts[i][j] / pts[i][3]);
            bboxmax[j] = std::max(bboxmax[j], pts[i][j] / pts[i][3]);
        }
    Vec2i P;
    TGAColor color;
    for (P.x = bboxmin.x; P.x <= bboxmax.x; P.x++)
        for (P.y = bboxmin.y; P.y <= bboxmax.y; P.y++) {
            Vec3f c = barycentric(project<2>(pts[0] / pts[0][3]), project<2>(pts[1] / pts[1][3]),
                                  project<2>(pts[2] / pts[2][3]),
                                  project<2>(P));
            // 深度值
            float z = pts[0][2] * c.x + pts[1][2] * c.y + pts[2][2] * c.z;
            float w = pts[0][3] * c.x + pts[1][3] * c.y + pts[2][3] * c.z;
            int frag_depth = std::max(0, std::min(255, int(z / w + 0.5)));
            if (c.x < 0 || c.y < 0 || c.z < 0 || zbuffer.get(P.x, P.y)[0] > frag_depth)
                continue;
            bool discard = shader.fragment(c, color);
            if (!discard) {
                zbuffer.set(P.x, P.y, TGAColor(frag_depth));
                image.set(P.x, P.y, color);
            }
        }
}