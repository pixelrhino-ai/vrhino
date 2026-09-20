#pragma once
#include <utility>
namespace vrhino {
// Construct before acquisition, capture the owner by reference, dismiss only
// after ownership publication. Cleanup must be noexcept (or fail closed).
template<class F> class ResourceRollback {
public:
    explicit ResourceRollback(F cleanup) : cleanup_(std::move(cleanup)) {}
    ResourceRollback(const ResourceRollback&)=delete;
    ResourceRollback& operator=(const ResourceRollback&)=delete;
    ~ResourceRollback() noexcept { if(active_) cleanup_(); }
    void commit() noexcept { active_=false; }
private:
    F cleanup_;
    bool active_=true;
};
template<class F> ResourceRollback(F)->ResourceRollback<F>;
} // namespace vrhino
