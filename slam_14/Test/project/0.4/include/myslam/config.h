//
// Created by zbox on 18-9-6.
//

#ifndef MYSLAM_CONFIG_H
#define MYSLAM_CONFIG_H

#include "myslam/common_include.h"

namespace myslam
{
    class Config
    {
    private:
        static std::shared_ptr<Config> config_;
        // why private and static, existed before creating class object,
        // so we can use this member directly without creating object
        cv::FileStorage file_;

        Config()    {}  // private constructor makes a singleton

    public:
        ~Config();  // close the file when deconstructing

        // set a new config file
        static void setParameterFile(const std::string& filename);
        // static member function can be called without creating class object

        // access the parameter values
        template <typename T>
        static T get(const std::string& key )
        {
            return T(config_->file_[key]);
        }
    };
}
#endif //MYSLAM_CONFIG_H