# create an executable
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PEG_INSERTION_BINARY_DIR}/01-kinesthetic-teaching)
ADD_EXECUTABLE (control_01_robot controller.cpp ${PEG_INSERTION_COMMON_SOURCE})

# and link the library against the executable
TARGET_LINK_LIBRARIES (control_01_robot ${PEG_INSERTION_COMMON_LIBRARIES})

# export resources such as model files.
# NOTE: this requires an install build
SET(APP_RESOURCE_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resources)
FILE(MAKE_DIRECTORY ${APP_RESOURCE_DIR})
FILE(COPY panda_arm.urdf DESTINATION ${APP_RESOURCE_DIR})
