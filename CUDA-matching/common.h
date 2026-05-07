#pragma once

#include "kseq/kseq.h"
#include <string>
#include <vector>

struct MatchResult
{
    std::string sample_name;
    std::string signature_name;
    double match_score;
    int integrity_hash;
};

void runMatcher(const std::vector<klibpp::KSeq> &samples,
                const std::vector<klibpp::KSeq> &signatures,
                std::vector<MatchResult> &matches);