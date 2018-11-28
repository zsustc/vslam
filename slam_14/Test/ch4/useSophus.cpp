#include<iostream>
#include<cmath>
using namespace std;

#include<Eigen/Core>
#include<Eigen/Geometry>

#include "sophus/so3.h"
#include "sophus/se3.h"

int main()
{
	Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d(0,0,1)).toRotationMatrix();
	
	Sophus::SO3 SO3_R(R);
	Sophus::SO3 SO3_v(0,0,M_PI/2);
	Eigen::Quaterniond q(R);
	Sophus::SO3 SO3_q(q);
	
	//output SO(3) with so(3) type
	cout << "SO(3) from matrix: " << SO3_R << endl;
	cout << "SO(3) from vector: " << SO3_v << endl;
	cout << "SO(3) from quaternion: " << SO3_q << endl;
	
	//using log map to get their li daishu
	Eigen::Vector3d so3 = SO3_R.log();
	cout << "so3 = " << so3.transpose() << endl;
	//hat is the symbol that means transform from vector to anti-symmetric matrix
	cout << "so3 hat = " << Sophus::SO3::hat(so3) << endl;
	cout << "so3 hat vee =" << Sophus::SO3::vee(Sophus::SO3::hat(so3)).transpose() << endl;
	
	Eigen::Vector3d update_so3(1e-4, 0, 0);
	Sophus::SO3 SO3_updated =Sophus::SO3::exp(update_so3) * SO3_R;
	cout << "SO3 updated = " << SO3_updated << endl;
	
	cout << "I am the cutting line" << endl;
	
	//operation to SE(3)
	Eigen::Vector3d t(1,0,0); //translation over axis X
	Sophus::SE3 SE3_Rt(R,t);
	Sophus::SE3 SE3_qt(q,t);
	cout << "SE3 from R,t = " << endl << SE3_Rt << endl;
	cout << "SE# from q,t = " << endl << SE3_qt << endl;
	
	typedef Eigen::Matrix<double,6,1> Vector6d;
	Vector6d se3 = SE3_Rt.log();
	cout << "se3 = " << se3.transpose() << endl;
	
	cout << "se3 hat = " << Sophus::SE3::hat(se3) << endl;
	cout << "se3 hat vee = " << Sophus::SE3::vee(Sophus::SE3::hat(se3)).transpose() << endl;
	
	Vector6d update_se3;
	update_se3.setZero();
	update_se3(0,0) = 1e-4d;
	Sophus::SE3 SE3_updated = Sophus::SE3::exp(update_se3) * SE3_Rt;
	cout << "SE3_ update = " << endl << SE3_updated.matrix() << endl;
	
	return 0;	
		
}