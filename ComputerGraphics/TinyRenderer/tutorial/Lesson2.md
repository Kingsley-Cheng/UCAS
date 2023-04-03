# Traingle Rasterization & back face culling
## Line Sweep Algorithm
通过对三角形每个 y 值横向扫描进行填充。
```C++
// 输入三角形的三个顶点
void triangle(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage& image, TGAColor color)
{
    // 将三角形的三个顶点根据 y 值大小进行排序
    if(t0.y > t1.y) std::swap(t0,t1);
    if(t0.y > t2.y) std::swap(t0, t2);
    if(t1.y > t2.y) std::swap(t1, t2);

    // 三角形的高度
    int total_height = t2.y - t0.y;
    // 将三角形根据 t1 点水平切割成两个三角形
    // 对下半三角形进行水平扫描
    for(int y = t0.y; y<= t1.y; y++){
        // 切割下半三角形高度
        int segment_height = t1.y - t0.y + 1;
        // 相似三角形比例
        float alpha = float(y-t0.y)/total_height;
        float beta = float(y-t0.y) / segment_height;
        // 水平扫描的两个边界点
        Vec2i A = t0 + (t2-t0) * alpha;
        Vec2i B = t0 + (t1-t0) * beta;
        // 判断哪一个在左边
        if(A.x > B.x) std::swap(A, B);
        for(int j = A.x; j<=B.x;j++){
            image.set(j, y, color);
        }
    }
    // 对上半个三角形做同样的操作
    for(int y = t1.y; y<=t2.y; y++){
        // 切割上半三角形高度
        int segment_height = t2.y - t1.y + 1;
        // 相似三角形比例
        float alpha = float (y-t0.y)/total_height;
        float beta = float (y - t1.y) / segment_height;
        // 水平扫描的两个边界点
        Vec2i A = t0 + (t2-t0) * alpha;
        Vec2i B = t1 + (t2 - t1) * beta;
        // 判断哪一个在左边
        if(A.x>B.x) std::swap(A, B);
        for(int j=A.x; j<=B.x; j++){
            image.set(j, y, color);
        }
    }
}

// main() 函数中
// 定义三角形三个顶点
Vec2i t0(5,7),t1(250,90),t2(100,400);
triangle(t0,t1,t2,image,red);
```
结果：
<img src="lesson2_1.png" alt="pic1">
注意到，放大观察可以发现三角形周边有出现走样现象。

## 包围盒 + 重心坐标
可以通过包围盒框出三角形所在的像素矩形框，再使用重心坐标来判断像素是否在三角形内，从而实现三角形的填充。
```C++
// 计算一个向量叉乘
Vec3f cross(Vec3f u, Vec3f v) {
    // (y*z-z*y,z*x-x*z,x*y-y*x)
    return Vec3f(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
}

// 计算 P 在三角形中的重心坐标
Vec3f barycentric(Vec2i *pts, Vec2i P) {
    // 重心坐标的权重与之垂直
    Vec3f u = cross(Vec3f(pts[2].x - pts[0].x, pts[1].x - pts[0].x, pts[0].x - P.x),
                    Vec3f(pts[2].y - pts[0].y, pts[1].y - pts[0].y, pts[0].y - P.y));
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

// main() 函数中
// 定义三角形三个顶点
    Vec2i pts[3] = {Vec2i(5, 7), Vec2i(250, 90), Vec2i(100, 400)};
    triangle(pts, image, red);
```
结果与 Line Sweep Algorithm 中一样。除了使用重心坐标外，我们还可以使用与三角形顶点的顺序叉乘来判断一个点是否在内部。

## Draw a model with random color
```C++
int main(int argc, char **argv) {
    // 创建环境
    int width = 500;
    int height = 500;
    TGAImage image(width, height, TGAImage::RGB);
    // 实例化模型
    Model model("../obj/african_head.obj");
    // 遍历模型的三角形面片
    for (int i = 0; i < model.nfaces(); i++) {
        std::vector<int> face = model.face(i);
        // 存放三角形的三个顶点
        Vec2i screen_coords[3];
        for (int j = 0; j < 3; j++) {
            // 做 viewport transform
            Vec3f world_coords = model.vert(face[j]);
            screen_coords[j] = Vec2i((world_coords.x + 1.) * width / 2.,
                                     (world_coords.y + 1.) * height / 2.);
        }
        triangle(screen_coords, image, TGAColor(rand() % 255, rand() % 255, rand() % 255, 255));
    }
    // 垂直翻转，把原点移到左下角
    image.flip_vertically();
    image.write_tga_file("output.tga");
    return 0;
}
```
<img src="lesson2_2.png" alt="pic2">

## Simple shading & lighting
我们下面加入简单的光照。
```C++
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
```
<img src="lesson2_3.png" alt="pic3">