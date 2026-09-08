#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "vrhino/product/progress.h"

namespace product = vrhino::product;

namespace {

void require_test(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

size_t count_character(const std::string& value, const char character) {
    return static_cast<size_t>(std::count(value.begin(), value.end(), character));
}

}  // namespace

int main() {
    try {
        std::ostringstream interactive_output;
        {
            product::ConsoleProgress progress(interactive_output, true);
            for (uint64_t completed = 0; completed <= 100; ++completed)
                progress.update("Acquiring source", completed, 100);
        }
        const std::string interactive = interactive_output.str();
        require_test(interactive.find('\r') != std::string::npos,
                     "interactive progress did not update in place");
        require_test(count_character(interactive, '\n') == 1,
                     "interactive progress emitted multiple terminal lines");
        require_test(interactive.find("\x1b") == std::string::npos,
                     "interactive progress emitted ANSI escapes");
        require_test(interactive.find("100%") != std::string::npos,
                     "interactive progress did not complete");

        std::ostringstream conversion_terminal;
        {
            product::ConsoleProgress progress(conversion_terminal, true);
            progress.update("Converting", 0, 4);
            progress.update("Converting", 1, 4);
            progress.update("Converting", 3, 4);
            progress.update("Converting", 4, 4);
        }
        require_test(count_character(conversion_terminal.str(), '\n') == 1 &&
                         conversion_terminal.str().find("\rConverting") !=
                             std::string::npos,
                     "interactive conversion progress did not stay on one line");

        std::ostringstream staged_terminal;
        {
            product::ConsoleProgress progress(staged_terminal, true);
            progress.update("Converting", 0, 2);
            progress.update("Converting", 2, 2);
            progress.update("Finalizing", 0, 2);
            progress.update("Finalizing", 2, 2);
        }
        require_test(count_character(staged_terminal.str(), '\n') == 2 &&
                         staged_terminal.str().find("\rFinalizing") !=
                             std::string::npos,
                     "conversion/finalization stages did not each use one line");

        std::ostringstream redirected_output;
        {
            product::ConsoleProgress progress(redirected_output, false);
            for (uint64_t completed = 0; completed <= 100; ++completed)
                progress.update("Acquiring source", completed, 100);
        }
        const std::string redirected = redirected_output.str();
        require_test(redirected.find('\r') == std::string::npos,
                     "redirected progress contains carriage returns");
        require_test(redirected.find("\x1b") == std::string::npos,
                     "redirected progress contains ANSI escapes");
        require_test(count_character(redirected, '\n') <= 11,
                     "redirected progress was not throttled");
        require_test(redirected.find("100%") != std::string::npos,
                     "redirected progress did not complete");

        std::ostringstream conversion_log;
        {
            product::ConsoleProgress progress(conversion_log, false);
            for (uint64_t completed = 0; completed <= 1000; ++completed)
                progress.update("Converting", completed, 1000);
        }
        require_test(conversion_log.str().find('\r') == std::string::npos &&
                         conversion_log.str().find("\x1b") == std::string::npos,
                     "redirected conversion progress contains terminal controls");
        require_test(count_character(conversion_log.str(), '\n') <= 11,
                     "redirected conversion progress was not throttled");

        std::cout << "native progress tests: PASS\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "native progress tests: FAIL: " << exception.what() << '\n';
        return 1;
    }
}
