#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        auto it = freeBlocks.begin();

        while (it != freeBlocks.end()) {
            if (it->second >= size) {
                size_t addr = it->first;

                if (it->second == size) {
                    // Properly fit
                    freeBlocks.erase(it);
                } else {
                    // Split free block
                    freeBlocks[addr + size] = it->second - size;
                    freeBlocks.erase(it);
                }

                used += size;
                if (used > peak) {
                    peak = used;
                }

                return addr;
            }
            ++it;
        }

        // No suitable free block
        size_t addr = used;     // new memory used at the end
        used += size;
        
        if (used > peak) {
            peak = used;
        }

        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        used -= size;

        // previous free block
        auto prevIt = freeBlocks.upper_bound(addr);
        if (prevIt != freeBlocks.begin()) {
            prevIt--;
        } else {
            prevIt = freeBlocks.end();
        }

        // next free block
        auto nextIt = freeBlocks.find(addr + size);

        size_t newAddr = addr;
        size_t newSize = size;

        if (prevIt != freeBlocks.end() && prevIt->first + prevIt->second == addr) {
            newAddr = prevIt->first;
            newSize += prevIt->second;
            freeBlocks.erase(prevIt);
        }

        if (nextIt != freeBlocks.end()) {
            newSize += nextIt->second;
            freeBlocks.erase(nextIt);
        }

        freeBlocks[newAddr] = newSize;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
