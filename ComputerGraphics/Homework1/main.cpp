#include "Triangle.hpp"
#include "rasterizer.hpp"
#include <Eigen/Core>
#include <iostream>
#include <opencv2/opencv.hpp>


constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0], 0, 1, 0, -eye_pos[1], 0, 0, 1,
        -eye_pos[2], 0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    model << std::cos(rotation_angle/180*MY_PI), -std::sin(rotation_angle/180*MY_PI), 0, 0, std::sin(rotation_angle/180*MY_PI), std::cos(rotation_angle/180*MY_PI), 0, 0, 0, 0, 1, 0,0,0,0,1;

    // TODO: Implement this function
    // Create the model matrix for rotating the triangle around the Z axis.
    // Then return it.

    return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{
    // Students will implement this function

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
    float x_near_height = std::tan(eye_fov/360*MY_PI) * zNear *2;
    float x_near_width = aspect_ratio * x_near_height;
    float x_far_height = std::tan(eye_fov/360*MY_PI) * zFar *2;
    float x_far_width = aspect_ratio * x_far_height;
    Eigen::Matrix4f persp2orth = Eigen::Matrix4f::Identity(); 
    Eigen::Matrix4f orth = Eigen::Matrix4f::Identity();
    persp2orth << zNear , 0 , 0 , 0 , 0 , zNear , 0 , 0 , 0 , 0 , zNear+zFar ,-zNear*zFar,0,0,1,0;
    orth <<2/x_near_width,0,0,0,0,2/x_near_height,0,0,0,0,2/(zFar-zNear),0,0,0,0,1;
    projection = orth * persp2orth;
    // TODO: Implement this function
    // Create the projection matrix for the given parameters.
    // Then return it.
    return projection;
}

Eigen::Matrix4f get_rotation(Vector3f point, Vector3f axis, float angle)
{
    Eigen::Matrix4f move2orign = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f move2orign_inv = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f model_axis = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f N = Eigen::Matrix3f::Identity();
    move2orign<<1,0,0,-point[0],0,1,0,-point[1],0,0,1,-point[2],0,0,0,1;
    move2orign_inv<<1,0,0,point[0],0,1,0,point[1],0,0,1,point[2],0,0,0,1;
    if(axis.norm() !=1)
        axis = axis / axis.norm();
    N << 0,-axis[2],axis[1], axis[2], 0, -axis[0], -axis[1], axis[0],0;
    model_axis.block<3,3>(0,0) = std::cos(angle/180*MY_PI) * Eigen::Matrix3f::Identity()+ (1-std::cos(angle/180*MY_PI))*axis*axis.transpose()+ std::sin(angle/180*MY_PI)*N;
    return (move2orign_inv * model_axis)* move2orign;
}


int main(int argc, const char** argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    Eigen::Vector3f axis(1.0f,2.0f,3.0f);
    Eigen::Vector3f point(1.0f,2.0f,3.0f);

    if (argc >= 3) {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4) {
            filename = std::string(argv[3]);
        }
        else
            return 0;
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0, 0, 5};

    std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};

    std::vector<Eigen::Vector3i> ind{{0, 1, 2}};

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key = 0;
    int frame_count = 0;

    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        // r.set_model(get_model_matrix(angle));
        r.set_model(get_rotation(point, axis, angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        // r.set_model(get_model_matrix(angle));
        r.set_model(get_rotation(point, axis, angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a') {
            angle += 10;
        }
        else if (key == 'd') {
            angle -= 10;
        }
    }

    return 0;
}
