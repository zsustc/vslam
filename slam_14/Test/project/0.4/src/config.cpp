//
// Created by zbox on 18-9-6.
//

#include "myslam/config.h"
namespace myslam
{
    void Config::setParameterFile(const std::string &filename)
    {
        if (nullptr == config_)
        {
            config_ = shared_ptr<Config> (new Config);
        }

        config_->file_ = cv::FileStorage(filename.c_str(), cv::FileStorage::READ);
        if (config_->file_.isOpened() == false) // open file fail and release file_
        {
            std::cerr << "parameter file " << filename << "does not exist. " << std::endl;
            config_->file_.release();
            return;
        }
    }

    Config::~Config()
    {
        if ( file_.isOpened())  // if file is still opened, it should be released after using
            file_.release();
    }

    shared_ptr<Config> Config::config_ = nullptr;
}
