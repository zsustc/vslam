//
// Created by zbox on 18-8-24.
//

#include "myslam/camera.h"

namespace myslam
{
    Camera::Camera()
    {
    }

    Vector2d Camera::camera2pixel(const Vector3d &p_c)
    {
        float u = fx_ * p_c(0,0) / p_c(2,0) + cx_;
        float v = fy_ * p_c(1,0) / p_c(2,0) + cy_;

        return Vector2d(u,v);
    }

    Vector3d Camera::world2camera(const Vector3d &p_w, const SE3 &T_c_w)
    {
        return T_c_w * p_w;
    }

    Vector3d Camera::camera2wolrd(const Vector3d &p_c, const SE3 &T_c_w)
    {
        return T_c_w.inverse() * p_c;
    }

    Vector3d Camera::pixel2camera(const Vector2d &p_p, double depth)
    {
        /*float x = depth * (p_p(0,0) - cx_) / fx_;
        float y = depth * (p_p(1,0) - cy_) / fy_;

        return Vector3d(x,y,depth);*/

        return Vector3d(
                depth * (p_p(0,0) - cx_) / fx_,
                depth * (p_p(1,0) - cy_) / fy_,
                depth
                );
    }

    Vector2d Camera::world2pixel(const Vector3d &p_w, const SE3 &T_c_w)
    {
        //return camera2pixel(T_c_w * p_w);
        return camera2pixel(world2camera(p_w, T_c_w));
    }

    Vector3d Camera::pixel2world(const Vector2d &p_p, const SE3 &T_c_w, double depth)
    {
        //return T_c_w.inverse() * pixel2camera(p_p, depth);
        return camera2wolrd(pixel2camera(p_p, depth), T_c_w);
    }
}

