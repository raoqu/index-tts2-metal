#include "mit2/allocator.hpp"

#include <limits>
#include <stdexcept>

namespace mit2 {

ScratchAllocator::ScratchAllocator(size_t capacity, size_t alignment)
    : capacity_(capacity), alignment_(alignment) {
    if (alignment_ == 0) {
        throw std::invalid_argument("alignment must be non-zero");
    }
}

size_t ScratchAllocator::align_up(size_t value) const {
    const size_t remainder = value % alignment_;
    if (remainder == 0) {
        return value;
    }
    const size_t padding = alignment_ - remainder;
    if (value > std::numeric_limits<size_t>::max() - padding) {
        throw std::overflow_error("scratch allocator alignment overflow");
    }
    return value + padding;
}

Allocation ScratchAllocator::allocate(const std::string& name, size_t size) {
    const size_t offset = align_up(used_);
    if (size > std::numeric_limits<size_t>::max() - offset) {
        throw std::overflow_error("scratch allocator allocation overflow");
    }
    const size_t end = offset + size;
    if (end > capacity_) {
        throw std::runtime_error("scratch allocator capacity exceeded");
    }
    used_ = end;
    if (used_ > peak_) {
        peak_ = used_;
    }
    Allocation alloc{name, offset, size};
    allocations_.push_back(alloc);
    return alloc;
}

void ScratchAllocator::rewind(size_t mark) {
    if (mark > used_) {
        throw std::invalid_argument("scratch allocator rewind mark is beyond used bytes");
    }
    while (!allocations_.empty() && allocations_.back().offset >= mark) {
        allocations_.pop_back();
    }
    if (!allocations_.empty()) {
        const auto& last = allocations_.back();
        const size_t last_end = last.offset + last.size;
        if (mark < last_end) {
            throw std::invalid_argument("scratch allocator rewind mark splits an allocation");
        }
    }
    used_ = mark;
}

void ScratchAllocator::reset() {
    used_ = 0;
    allocations_.clear();
}

}  // namespace mit2
