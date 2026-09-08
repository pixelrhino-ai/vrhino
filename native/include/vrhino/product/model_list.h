#pragma once

#include <string>
#include <vector>

#include "vrhino/json.h"
#include "vrhino/product/model_package.h"

namespace vrhino::product {

struct LocalModelSummary {
    std::string reference;
    std::string name_space;
    std::string name;
    std::string version;
    std::string architecture;
    std::string product_family;
    bool installed = true;
};

std::vector<LocalModelSummary> list_local_models(const LocalModelCache& cache);
Json build_local_model_list(const LocalModelCache& cache);

}  // namespace vrhino::product
