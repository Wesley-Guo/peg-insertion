# create an executable
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PEG_INSERTION_BINARY_DIR}/00-dual_proxy_motion_normal_force_robot)
ADD_EXECUTABLE (control_00_robot controller.cpp ${PEG_INSERTION_COMMON_SOURCE})
ADD_EXECUTABLE (simviz_00_robot simviz.cpp ${PEG_INSERTION_COMMON_SOURCE})
ADD_EXECUTABLE (viz_00_robot viz.cpp ${PEG_INSERTION_COMMON_SOURCE})

# and link the library against the executable
TARGET_LINK_LIBRARIES (control_00_robot ${PEG_INSERTION_COMMON_LIBRARIES})
TARGET_LINK_LIBRARIES (simviz_00_robot ${PEG_INSERTION_COMMON_LIBRARIES})
TARGET_LINK_LIBRARIES (viz_00_robot ${PEG_INSERTION_COMMON_LIBRARIES})

# export resources such as model files.
# NOTE: this requires an install build
SET(APP_RESOURCE_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resources)
FILE(MAKE_DIRECTORY ${APP_RESOURCE_DIR})
FILE(COPY world_probe.urdf DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY panda_arm.urdf DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY panda_arm_ati_probe.urdf DESTINATION ${APP_RESOURCE_DIR})
