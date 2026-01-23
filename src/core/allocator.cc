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
        // IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        // 安全处理0字节分配
        if (size == 0) {
            return 0;
        }

        // 使用first-fit策略分配内存
        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ) {
            size_t blockStart = it->first;
            size_t blockSize = it->second;
            
            if (blockSize >= size) {
                size_t allocatedStart = blockStart;
                it = freeBlocks.erase(it);
                
                if (blockSize > size) {
                    freeBlocks[blockStart + size] = blockSize - size;
                }
                
                this->used += size;
                if (this->used > this->peak) this->peak = this->used;
                
                return allocatedStart;
            } else {
                ++it;
            }
        }
        
        // 没有找到合适的free block，分配新的内存
        size_t newAddr = this->used;
        this->used += size;
        if (this->used > this->peak) this->peak = this->used;
        
        return newAddr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        // IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        if (size == 0) {
            return;
        }
        
        freeBlocks[addr] = size;
        
        // 合并相邻blocks
        bool merged;
        do {
            merged = false;
            for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it) {
                auto next = std::next(it);
                if (next != freeBlocks.end() && it->first + it->second == next->first) {
                    size_t mergedSize = it->second + next->second;
                    size_t mergedStart = it->first;
                    
                    auto next_it = std::next(it);
                    freeBlocks.erase(next_it);
                    it = freeBlocks.erase(it);
                    
                    freeBlocks[mergedStart] = mergedSize;
                    merged = true;
                    break;
                }
            }
        } while (merged);
        
        if (this->used >= size) {
            this->used -= size;
        } else {
            this->used = 0;
        }
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
