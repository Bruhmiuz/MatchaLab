// YOU MAY EDIT THIS FILE
#pragma once

#include <vector>
#include <utility>
#include <string>
#include "common.h"

class Engine {
public:
    std::vector<DataPoint> dataPoint;

    void KNN(Params &p, std::vector<DataPoint> &dataset, std::vector<Query> &queries);
};
