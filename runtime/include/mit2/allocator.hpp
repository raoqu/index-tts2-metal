#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mit2 {

struct Allocation {
    std::string name;
    size_t offset = 0;
    size_t size = 0;
};

class ScratchAllocator {
public:
    explicit ScratchAllocator(size_t capacity, size_t alignment = 256);

    Allocation allocate(const std::string& name, size_t size);
    size_t checkpoint() const { return used_; }
    void rewind(size_t mark);
    void reset();

    size_t capacity() const { return capacity_; }
    size_t used() const { return used_; }
    size_t peak() const { return peak_; }
    size_t alignment() const { return alignment_; }
    const std::vector<Allocation>& allocations() const { return allocations_; }

private:
    size_t align_up(size_t value) const;

    size_t capacity_;
    size_t alignment_;
    size_t used_ = 0;
    size_t peak_ = 0;
    std::vector<Allocation> allocations_;
};

}  // namespace mit2
