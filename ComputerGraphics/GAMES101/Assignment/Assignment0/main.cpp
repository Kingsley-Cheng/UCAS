#include<iostream>
#include<cmath>
#include<Eigen/Core>

int main(){
    // Basic Code
    float a=1.0, b=2.0;
    std::cout<<"----Basic Code----"<<std::endl;
    std::cout<<a<<std::endl;
    std::cout<<a/b<<std::endl;
    std::cout<<std::sqrt(a)<<std::endl;
    std::cout<<std::acos(-1)<<std::endl;
    std::cout<<std::sin(30.0/180.0*acos(-1))<<std::endl;
    
    // Eigen Pack
    std::cout<<std::endl;
    std::cout<<"----Eigen Pack----"<<std::endl;

    // Example of vector
    std::cout<<"Example of Vector \n";
    // vector definition
    Eigen::Vector3f v(1.0f, 2.0f, 3.0f);
    Eigen::Vector3f w(1.0f, 0.0f, 0.0f);
    // vector output
    std::cout<<"Example of output \n";
    std::cout<<v<<std::endl;
    // vector add
    std::cout<<"Example of add \n";
    std::cout<<v+w<<std::endl;
    // vector scalar multiply
    std::cout<<"Example of scalar multiply \n";
    std::cout<< v*3.0f<<std::endl;
    std::cout<< 2.0f*v<<std::endl;

    // Example of matrix
    std::cout<<"Example of matrix \n";
    // matrix definition
    Eigen::Matrix3f i,j;
    i<<1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0;
    j<<2.0,3.0,1.0,4.0,6.0,5.0,9.0,7.0,8.0;
    // matrix output
    std::cout<< "Example of output \n";
    std::cout<< i <<std::endl;
    // matrix add i + j
    std::cout<< "Example of matrix add i + j"<<std::endl;
    std::cout<<i+j<<std::endl;
    // matrix scalar multiply
    std::cout<< "Example of matrix scalar multiply i * 2.0"<<std::endl;
    std::cout<<i* 2.0f<<std::endl;
    // matrix multiply i * j
    std::cout<< "Example of matrix multiply i * j"<<std::endl;
    std::cout<<i*j<<std::endl;
    // matrix multiply vector i * v
    std::cout<< "Example of matrix multiply vector i * v"<<std::endl;
    std::cout<<i*v<<std::endl;

    // Transform
    std::cout<< "\n----Homework of Transform----"<<std::endl;
    Eigen::Vector3f p(2,1,1);
    Eigen::Matrix3f T;
    T<<std::sqrt(2)/2,-std::sqrt(2)/2,1,std::sqrt(2)/2,std::sqrt(2)/2,2,0,0,1;
    std::cout<<"Vector:\n"<<p<<std::endl;
    std::cout<<"Transform Matrix:\n"<<T<<std::endl;
    std::cout<<"After Transformation:"<<std::endl;
    std::cout<<T*p<<std::endl;
    return 0;
}