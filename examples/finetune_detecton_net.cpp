// Copyright 2013 Ross Girshick
//
// This is a simple script that allows one to quickly finetune a network
// for detection.
//
// Based on finetune_net.cpp by Yangqing Jia.
//
// Usage:
//    finetune_detection_net solver_proto_file pretrained_net

#include <cuda_runtime.h>

#include <cstring>

#include "caffe/caffe.hpp"

using namespace caffe;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 2) {
    LOG(ERROR) << "Usage: finetune_detection_net solver_proto_file pretrained_net";
    return 0;
  }

  SolverParameter solver_param;
  ReadProtoFromTextFile(argv[1], &solver_param);

  LOG(INFO) << "Starting optimization";
  SGDSolver<float> solver(solver_param);
  LOG(INFO) << "Loading from " << argv[2];

  NetParameter pretrained_net_param;
  ReadProtoFromBinaryFile(string(argv[2]), &pretrained_net_param);

  solver.net()->CopyTrainedLayersFrom(pretrained_net_param);
  solver.Solve();
  LOG(INFO) << "Optimization Done.";

  return 0;
}
