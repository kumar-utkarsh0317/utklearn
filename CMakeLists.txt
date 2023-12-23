cmake_minimum_required(VERSION 3.22.1)

project(utkarsh_library)

add_library(utklearn src/train_test_split.cpp)

# Add the header directory to the include path
include_directories(header)

# set_target_properties(target
#     PROPERTIES
#     property1 value1
#     property2 value2
#     ...
# )
# target: This is the name of the target for which you want to set properties. Targets can be executables, libraries, or custom targets
# PROPERTIES: This keyword indicates that you are about to specify properties for the target.
# when you want to install the library and make its headers accessible for other projects
set_target_properties(utklearn PROPERTIES PUBLIC_HEADER header/utklearn.hpp)

install(TARGETS utklearn LIBRARY DESTINATION lib 
        PUBLIC_HEADER DESTINATION include
        )