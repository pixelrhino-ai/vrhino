#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>

namespace vrhino::product {

class ConsoleProgress {
public:
    ConsoleProgress(std::ostream& output, bool interactive);
    ~ConsoleProgress();

    void update(const std::string& stage, uint64_t completed, uint64_t total);
    void finish();

private:
    std::string render(const std::string& stage, uint64_t completed,
                       uint64_t total) const;

    std::ostream& output_;
    bool interactive_ = false;
    bool active_ = false;
    std::string stage_;
    int last_percent_ = -1;
    size_t last_width_ = 0;
};

}  // namespace vrhino::product
