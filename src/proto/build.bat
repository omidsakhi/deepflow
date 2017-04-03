@echo off
protoc deepflow.proto --cpp_out=.
move "./deepflow.pb.h" "../../include/proto/deepflow.pb.h"