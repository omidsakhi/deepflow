@echo off
cls
cd..
cd third-party
cd socket.io-client-cpp
mkdir cmake-build
cd cmake-build
cmake -G "Visual Studio 14 2015 Win64" -DBOOST_ROOT:STRING="${CMAKE_CURRENT_SOURCE_DIR}/../../boost_1_64_0" -DBOOST_VER:STRING="1.64.0" ../