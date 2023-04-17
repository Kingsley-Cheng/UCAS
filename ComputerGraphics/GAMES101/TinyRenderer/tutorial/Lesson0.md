# Lession0 Project Setup
创建一个 C++ 的 Project, 名字为 tinyRenderer, 并把[源代码](https://github.com/ssloy/tinyrenderer)中的 `tgaimage.h` 和 `tgaimage.cpp` 拖入 Project 中。在 Project 中创建一个  `main.cpp` 文件，下面我们实现 “创建一个 100 $\times$ 100 的环境，并在（52，41）处画上一个红点” 这个功能。
```C++
#include "tgaimage.h"

// 定义红色与白色两种颜色
const TGAColor white = TGAColor(255, 255, 255, 255);// TGAColor 四个参数：红，绿，蓝，透明度
const TGAColor red = TGAColor(255,0,0,255);

int main(int argc, char** argv){
    TGAImage image(100,100,TGAImage::RGB); // 创建环境，100*100，RGB颜色通道
    image.set(52, 41, red); // 在（52，41）处画红点
    //为了使坐标原点在左下角，我们需要垂直翻转屏幕
    image.flip_vertically();
    // 存储在 "output.tga" 文件中
    image.write_tga_file("output.tga");
    return 0;
}
```

同时，你需要创建一个 `CMakeLists.txt` 文件，并在文件配置必要的 Cmake 环境。
```CMake
cmake_minimum_required(VERSION 3.0.0)
project(tinyRenderer VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(tinyRenderer main.cpp tgaimage.cpp tgaimage.h)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
```

接下来就需要对 Project 进行编译了，你需要在命令行进行如下操作(基于macOS)。
```shell
mkdir build
cd build
cmake ..
make
./tinyRenderer
```
如果没有问题，你将看到如下的结果：

<img src="Lession0.png" alt="pic1" style="zoom:150%;" >
