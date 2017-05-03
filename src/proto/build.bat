@echo off
protoc deepflow.proto --cpp_out=.
move "./deepflow.pb.h" "../../include/proto/deepflow.pb.h"
protoc caffe.proto --cpp_out=.
move "./caffe.pb.h" "../../include/proto/caffe.pb.h"