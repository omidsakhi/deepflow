@echo off
cls
cd..
cd third-party
cd boost_1_64_0
./bootstrap.bat
./b2 --toolset=msvc-14.0 architecture=x86 address-model=64 stage