#include <windows.h>
#include <cwchar>
int wmain(int argc, wchar_t** argv) {
    if (argc != 2 || std::wcscmp(argv[1], L"-version") != 0) return 2;
    constexpr char message[] = "VRhino Phase 7 synthetic helper; no media processing\n";
    DWORD count = 0;
    return WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), message, sizeof(message) - 1, &count, nullptr) ? 0 : 3;
}
