#include "vrhino/product/progress.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <ostream>
#include <sstream>

namespace vrhino::product {
namespace {

std::string format_bytes(const uint64_t bytes) {
    static constexpr std::array<const char*, 5> units = {
        "B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < units.size()) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream output;
    output << std::fixed << std::setprecision(unit == 0 ? 0 : 2)
           << value << ' ' << units[unit];
    return output.str();
}

}  // namespace

ConsoleProgress::ConsoleProgress(std::ostream& output, const bool interactive)
    : output_(output), interactive_(interactive) {}

ConsoleProgress::~ConsoleProgress() { finish(); }

std::string ConsoleProgress::render(const std::string& stage,
                                    const uint64_t completed,
                                    const uint64_t total) const {
    const int percent = total == 0
        ? 100
        : static_cast<int>(100 * std::min(completed, total) / total);
    std::ostringstream line;
    line << stage << "  " << format_bytes(completed) << " / "
         << format_bytes(total);
    if (interactive_) {
        constexpr int width = 20;
        const int filled = std::clamp(percent * width / 100, 0, width);
        line << "  [";
        for (int index = 0; index < width; ++index)
            line << (index < filled ? '=' : '.');
        line << "] " << percent << '%';
    } else {
        line << " (" << percent << "%)";
    }
    return line.str();
}

void ConsoleProgress::update(const std::string& stage, uint64_t completed,
                             const uint64_t total) {
    if (stage.empty()) return;
    completed = std::min(completed, total);
    const int percent = total == 0
        ? 100
        : static_cast<int>(100 * completed / total);
    if (active_ && stage != stage_) finish();
    if (!active_) {
        active_ = true;
        stage_ = stage;
        last_percent_ = -1;
        last_width_ = 0;
    }
    const bool complete = completed == total;
    const bool render_now = interactive_
        ? last_percent_ < 0 || percent > last_percent_ || complete
        : last_percent_ < 0 || percent >= last_percent_ + 10 || complete;
    if (!render_now) return;

    const std::string line = render(stage, completed, total);
    if (interactive_) {
        output_ << '\r' << line;
        if (line.size() < last_width_)
            output_ << std::string(last_width_ - line.size(), ' ');
        output_.flush();
        last_width_ = line.size();
        if (complete) output_ << '\n';
    } else {
        output_ << line << '\n';
    }
    last_percent_ = percent;
    if (complete) {
        active_ = false;
        stage_.clear();
        last_width_ = 0;
    }
}

void ConsoleProgress::finish() {
    if (!active_) return;
    if (interactive_) output_ << '\n';
    output_.flush();
    active_ = false;
    stage_.clear();
    last_percent_ = -1;
    last_width_ = 0;
}

}  // namespace vrhino::product
