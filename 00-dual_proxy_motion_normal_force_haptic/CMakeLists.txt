# create an executable
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PEG_INSERTION_BINARY_DIR}/00-dual_proxy_motion_normal_force_haptic)
ADD_EXECUTABLE (control_haptic controller.cpp ${PEG_INSERTION_COMMON_SOURCE})

# and link the library against the executable
TARGET_LINK_LIBRARIES (control_haptic ${PEG_INSERTION_COMMON_LIBRARIES})
