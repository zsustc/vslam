add_library(myslam SHARED
        frame.cpp
        map.cpp
        mappoint.cpp
        camera.cpp
        config.cpp
        visual_odometry.cpp
        g2o_types.cpp
        )

target_link_libraries(myslam
        ${THIRD_PARTY_LIBS}
        )