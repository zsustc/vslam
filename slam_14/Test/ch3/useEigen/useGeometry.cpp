#include<iostream>
#include<cmath>
using namespace std;

#include<Eigen/Core>

//Eigen geometry part
#include<Eigen/Geometry>

/******************************************************
this program shows how to use geometry part of Eigen
*******************************************************/

int main()
{
	Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
	
	Eigen::AngleAxisd rotation_vector (M_PI/4, Eigen::Vector3d(0,0,1));
	cout.precision(3);
	
	
	cout<<"rotation matrix =\n" << rotation_vector.matrix() << endl;
	rotation_matrix = rotation_vector.toRotationMatrix(); // rotation_vector can be transformed into rotation_matrix
	
	
	Eigen::Vector3d v(1,0,0);
	Eigen::Vector3d v_rotated = rotation_vector * v; // rotating vector v by rotation vector
	cout << "(1,0,0) after rotation =" << v_rotated.transpose() << endl;
	
	v_rotated = rotation_matrix * v;
	cout << "(1,0,0) after rotation" << v_rotated.transpose() << endl;
	
	//Eular angle, 
	Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2,1,0); //ZYX
	cout << "yaw pitch roll =" << euler_angles.transpose() << endl;
	
	//Isometry transformation by using Eigen::Isometry
	Eigen::Isometry3d T = Eigen::Isometry3d::Identity(); //size 4 * 4
	
	T.rotate(rotation_vector);
	T.pretranslate(Eigen::Vector3d(1,3,4));
	cout << "Trnsform matrix = \n" << T.matrix() << endl;
	
	Eigen::Vector3d v_transformed = T * v; // R * v + t
	cout << "v transformed =" << v_transformed.transpose() << endl;
	
	//Affine transformation Eigen::Affine3d and projective transformation Eigen::Projective3d
	
	//quaternion
	Eigen::Quaterniond q = Eigen::Quaterniond(rotation_vector);
	cout << "quaternion = \n" << q.coeffs() << endl; // (x,y,z,w)
	
	//q = Eigen::Quaterniond(rotation_matrix);
	cout << "quaternion = \n" << q.coeffs() << endl;
	
	v_rotated = q * v;//qvq^{-1}
	cout << "(1,0,0) after rotation =" << v_rotated.transpose() << endl;
	
	return 0;
	
}


