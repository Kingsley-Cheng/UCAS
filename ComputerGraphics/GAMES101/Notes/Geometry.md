# Geometry
## Representation of Geometry
### Implict
无明确表示，如用函数表示曲面，点(x, y, z)满足一定函数f(x, y)关系就在一个曲面上。判断点的位置关系很方便，遍历绘制图形比较困难。
- algebraic surface
- Constructive solid geometry(简单几何体的表达组合)
- level sets
- distance functions(常用于模型融合的模拟过程)
- fractals(分形)

Pros
- compact description 
- certain queries easy (inside object, distance to surface)
- good for ray-to-surface intersection
- for simple shapes, exact description / no sampling error
- easy to handle changes in topology 
Cons
- difficult to model complex shapes


### Explict
有明确表示方法，直接给出，或通过参数映射给出几何信息，比如一般的点云或网格。遍历绘制图形比较方便， 但判断点的位置关系，如内外、是否在表面上比较困难。

- point cloud(三维点坐标的集合)
- polygon mesh
  >.obj文件：
  >
  >v (顶点xyz坐标)
  >
  >vt（纹理坐标）
  >
  >vn（法线方向）
  >
  >f (顶点1坐标/ 顶点1纹理坐标 / 顶点1法线方向 顶点2... 顶点3...)
- subdivision, NURBS

## Curves & Surfaces
### $B\'ezier$ Curves
定义四个控制点，其中包含起点(p0)和终点(p3)，可以得到一条光滑的曲线（贝塞尔曲线）。

Properties:
- Interpolates endpoints
  > For cubic Bézier: $b(0) = b_0 ; b(1) = b_3$
- Tangent to end segments
  > Cubic case:$b'(0) = 3(b_1 - b_0 ); b'(1) = 3(b_3 -b_2)$
- Affine transformation property
  > Transform curve by transforming control points
- Convex hull property
  > Curve is within convex hull of control points

### de Casteljau Algorithm
对每个时间t，(n-1)次递归找控制点连接线的t分段位置点，n是控制点数量。最后可以总结为多项式（伯恩斯坦多项式）加权组合形式：
$$
b^n(t) = \sum\limits_{j=0}^nb_jB_j^n(t)
$$
其中
$$
B_j^n(t) = {n\choose i}t^j(1-t)^{n-i}
$$

### Piecewise Bézier Curves
1. 高阶贝塞尔曲线递归求解比较麻烦，所以一般采用分段定义，每段一般4个控制点组成。
2. 分段连接位置要保证光滑($C^1$)，只需要保证连接点处的切线矢量相等 （方向+长度）即可。

### Other types of splines
样条(spline): 由关键点组成的可控曲线（基函数控制），相比贝塞尔曲线有局部可调整性，不过略复杂
B-spline:
- Short for basis splines
- Require more information than Bezier curves
- Satisfy all important properties that Bézier curves have (i.e. superset)

### Bézier Surface
$4\times 4$ 控制点在两个方向生成 Bézier Curves

## Mesh Operation
### subdivision
-  Loop Subdivision(只针对三角网格)
> new vertices: 3/8 * (A + B) + 1/8 * (C + D)
>
> old vertices: Update to:(1 - n*u) * original_position + u * neighbor_position_sum
>
> n: vertex degress
> if n=3 u:3/16 else 3/(8n)

- Catmull Subdivision（通用任何网格）
> Quad-face、Non-quad-face、Extraordinary vertex(degree != 4)
>
> 每个面中间取一个点（可以是中心也可以是重心之类的）加入新点集，每个边的中间点加入新点集，然后把新点集中的点连接在一块儿。经过一次此操作，奇异点数量增加，非四边形面消失，因此后面奇异点书不再会增加。
> 
> 对新旧顶点进行调整。边上的新点是边起点、边止点、边两侧的面重心的平均；面上的新点是面顶点的平均；旧点调整相对复杂，如下：

### simplification
- edge collapsing
> Quadric Error Metrics: new vertex should minimize its sum of squaredistance (L2 distance) to previously related triangle planes!
>
> 按照每条边的二次度量误差对边建堆
>
> 坍缩最小二次误差的边，更新坍缩边周围的边的二次度量误差，调整堆不断循环上两步。
  