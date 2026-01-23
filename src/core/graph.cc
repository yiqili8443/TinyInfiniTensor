#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    /************* self-defined interfaces ********************/
    bool GraphObj::isInversePermutation(const vector<int> &perm1, const vector<int> &perm2) {
        if (perm1.size() != perm2.size()) {
            return false;
        }

        for (size_t i = 0; i < perm1.size(); i++) {
            if (perm2[perm1[i]] != static_cast<int>(i)) {
                return false;
            }
        }

        return true;
    }

    void GraphObj::cleanupWeakReferences() {
        std::cout << "=== Cleaning up weak references in graph ===" << std::endl;
    
        // 1. 构建有效的算子集合（O(1) 查找）
        std::unordered_set<Operator> valid_ops;
        for (const auto& op : ops) {
            if (op) valid_ops.insert(op);
        }
        
        // 2. 清理所有算子的前驱/后继关系
        for (auto& op : ops) {
            if (!op) continue;
            
            // 清理前驱
            std::vector<WRef<OperatorObj>> validPreds;
            for (auto& weakRef : op->predecessors) {
                if (auto sp = weakRef.lock()) {
                    if (valid_ops.find(sp) != valid_ops.end()) {
                        validPreds.push_back(weakRef);
                    }
                }
            }
            op->predecessors = validPreds;
            
            // 清理后继
            std::vector<WRef<OperatorObj>> validSuccs;
            for (auto& weakRef : op->successors) {
                if (auto sp = weakRef.lock()) {
                    if (valid_ops.find(sp) != valid_ops.end()) {
                        validSuccs.push_back(weakRef);
                    }
                }
            }
            op->successors = validSuccs;
            
            std::cout << "  Cleaned operator " << op->getGuid() 
                    << ": pred=" << validPreds.size() 
                    << ", succ=" << validSuccs.size() << std::endl;
        }
        
        // 3. 清理所有tensor的弱引用
        std::cout << "=== Cleaning tensor weak references ===" << std::endl;
        
        for (auto& tensor : tensors) {
            if (!tensor) continue;
            
            // 3.1 清理tensor的source弱引用
            auto source = tensor->getSource();
            if (source && valid_ops.find(source) == valid_ops.end()) {
                std::cout << "  Cleaning invalid source for tensor " << tensor->getGuid() 
                        << ": operator " << source->getGuid() << " not in graph" << std::endl;
                tensor->setSource(nullptr);  // 需要实现这个方法
            }
            
            // 3.2 清理tensor的targets弱引用
            std::vector<WRef<OperatorObj>> validTargets;
            for (auto& weakRef : tensor->targets) {
                if (auto sp = weakRef.lock()) {
                    if (valid_ops.find(sp) != valid_ops.end()) {
                        validTargets.push_back(weakRef);
                    } else {
                        std::cout << "  Cleaning invalid target for tensor " << tensor->getGuid() 
                                << ": operator " << sp->getGuid() << " not in graph" << std::endl;
                    }
                } else {
                    std::cout << "  Cleaning expired weak reference for tensor " << tensor->getGuid() << std::endl;
                }
            }
            tensor->targets = validTargets;
            
            if (tensor->targets.size() != validTargets.size()) {
                std::cout << "  Cleaned tensor " << tensor->getGuid() 
                        << ": targets " << tensor->targets.size() << " -> " << validTargets.size() << std::endl;
            }
        }

        // 4. 清理悬空tensor
        std::cout << "=== Removing dangling tensors ===" << std::endl;
        auto original_tensor_count = tensors.size();
        tensors.erase(
            std::remove_if(tensors.begin(), tensors.end(),
                [&](const Tensor& tensor) {
                    if (!tensor) return true;
                    
                    // 悬空tensor条件：没有source且没有targets
                    bool is_dangling = (!tensor->getSource() && tensor->getTargets().empty());
                    
                    if (is_dangling) {
                        std::cout << "  Removing dangling tensor " << tensor->getGuid() 
                                << ": no source and no targets" << std::endl;
                        
                        // 同时从所有算子的输入/输出中移除这个tensor
                        for (auto& op : ops) {
                            if (!op) continue;
                            
                            // 从输入中移除
                            op->inputs.erase(
                                std::remove(op->inputs.begin(), op->inputs.end(), tensor),
                                op->inputs.end()
                            );
                            
                            // 从输出中移除
                            op->outputs.erase(
                                std::remove(op->outputs.begin(), op->outputs.end(), tensor),
                                op->outputs.end()
                            );
                        }
                        
                        return true;
                    }
                    return false;
                }),
            tensors.end()
        );
        
        std::cout << "  Removed " << (original_tensor_count - tensors.size()) 
                << " dangling tensors" << std::endl;
        
        // 5. 验证连接关系
        std::cout << "=== Validating graph connections ===" << std::endl;
        
        // 5.1 验证算子输入/输出与tensor连接的一致性
        for (auto& op : ops) {
            if (!op) continue;
            
            // 验证输入tensor
            for (size_t i = 0; i < op->getInputs().size(); ++i) {
                auto input = op->getInputs()[i];
                if (!input) {
                    std::cout << "WARNING: Operator " << op->getGuid() 
                            << " has null input at index " << i << std::endl;
                    continue;
                }
                
                // 检查input的targets是否包含此算子
                bool found = false;
                for (auto& weakRef : input->targets) {
                    if (auto sp = weakRef.lock()) {
                        if (sp == op) {
                            found = true;
                            break;
                        }
                    }
                }
                
                if (!found) {
                    std::cout << "FIX: Adding missing target for tensor " << input->getGuid() 
                            << " -> operator " << op->getGuid() << std::endl;
                    input->targets.push_back(op);  // 简化版，实际需要检查重复
                }
            }
            
            // 验证输出tensor
            for (size_t i = 0; i < op->getOutputs().size(); ++i) {
                auto output = op->getOutputs()[i];
                if (!output) continue;
                
                // 检查output的source是否为此算子
                auto source = output->getSource();
                if (!source || source != op) {
                    std::cout << "FIX: Setting source for tensor " << output->getGuid() 
                            << " to operator " << op->getGuid() << std::endl;
                    output->setSource(op);  // 需要实现这个方法
                }
            }
        }
        
        std::cout << "=== Weak reference cleanup completed ===" << std::endl;
    }

    void GraphObj::eliminateTransposePair(Operator op1, Operator op2) {
        std::cout << "=== Eliminating transpose pair: " 
              << op1->getOpType().toString() << "(" << op1->getGuid() << ") and "
              << op2->getOpType().toString() << "(" << op2->getGuid() << ")" << std::endl;
    
        // 获取关键tensor
        Tensor inputTensor = op1->getInputs(0);    // tensor2
        Tensor middleTensor = op1->getOutput();    // tensor4 (也是 op2->getInputs(0))
        Tensor outputTensor = op2->getOutput();    // tensor5
        
        std::cout << "Tensors: input=" << (inputTensor ? inputTensor->getGuid() : -1)
                << ", middle=" << (middleTensor ? middleTensor->getGuid() : -1)
                << ", output=" << (outputTensor ? outputTensor->getGuid() : -1) << std::endl;
        
        if (!inputTensor || !middleTensor || !outputTensor) {
            std::cout << "ERROR: null tensors in eliminateTransposePair" << std::endl;
            return;
        }
        
        // 1. 获取所有需要重新连接的消费者
        std::vector<Operator> consumers;
        for (auto& target : outputTensor->getTargets()) {
            if (target) consumers.push_back(target);
        }
        
        std::cout << "Found " << consumers.size() << " consumers to reconnect" << std::endl;
        
        // 2. 重建连接关系 - 先更新所有消费者
        for (auto& consumer : consumers) {
            std::cout << "Processing consumer: " << consumer->getOpType().toString()
                    << "(" << consumer->getGuid() << ")" << std::endl;
            
            // 在修改前保存当前状态用于调试
            std::cout << "  Consumer inputs before:";
            for (auto& inp : consumer->getInputs()) {
                std::cout << " " << (inp ? inp->getGuid() : -1);
            }
            std::cout << std::endl;
            
            // 替换consumer的输入：将outputTensor替换为inputTensor
            bool replaced = false;
            for (size_t i = 0; i < consumer->getInputs().size(); ++i) {
                if (consumer->getInputs()[i] == outputTensor) {
                    std::cout << "  Replacing input at index " << i 
                            << " from tensor " << outputTensor->getGuid()
                            << " to tensor " << inputTensor->getGuid() << std::endl;
                    
                    // 重要：先移除旧连接，再添加新连接
                    outputTensor->removeTarget(consumer);
                    inputTensor->addTarget(consumer);
                    
                    // 直接修改inputs向量
                    consumer->inputs[i] = inputTensor;
                    replaced = true;
                }
            }
            
            if (replaced) {
                std::cout << "  Consumer inputs after:";
                for (auto& inp : consumer->getInputs()) {
                    std::cout << " " << (inp ? inp->getGuid() : -1);
                }
                std::cout << std::endl;
            }
        }
        
        // 3. 清理op1和op2之间的连接
        std::cout << "Cleaning connections between op1 and op2" << std::endl;
        middleTensor->setSource(nullptr);  // 清理tensor4的source
        middleTensor->removeTarget(op2);  // 清理tensor4的target
        
        // 4. 更新inputTensor的targets
        std::cout << "Updating inputTensor targets" << std::endl;
        inputTensor->removeTarget(op1);  // 移除指向op1的连接
        
        // 5. 清理op1和op2的前驱/后继关系
        std::cout << "Cleaning predecessor/successor relationships" << std::endl;
        
        // 清理op1的后继（op2）
        for (auto& succ : op1->getSuccessors()) {
            if (succ && succ == op2) {
                op1->removeSuccessors(op2);
                op2->removePredecessors(op1);
                break;
            }
        }
        
        // 6. 安全删除算子 - 先清理所有外部引用
        std::cout << "Removing operators from graph" << std::endl;
        
        // 从图中移除算子前，清理所有外部引用
        auto it_op1 = std::find(ops.begin(), ops.end(), op1);
        if (it_op1 != ops.end()) {
            ops.erase(it_op1);
        }
        
        auto it_op2 = std::find(ops.begin(), ops.end(), op2);
        if (it_op2 != ops.end()) {
            ops.erase(it_op2);
        }
        
        // 7. 删除tensor - 检查引用计数
        std::cout << "Removing tensors" << std::endl;
        
        // 删除middleTensor (tensor4)
        if (middleTensor->getTargets().empty() && !middleTensor->getSource()) {
            auto it_mid = std::find(tensors.begin(), tensors.end(), middleTensor);
            if (it_mid != tensors.end()) {
                tensors.erase(it_mid);
                std::cout << "  Removed middle tensor " << middleTensor->getGuid() << std::endl;
            }
        }
        
        // 删除outputTensor (tensor5)
        if (outputTensor->getTargets().empty() && !outputTensor->getSource()) {
            auto it_out = std::find(tensors.begin(), tensors.end(), outputTensor);
            if (it_out != tensors.end()) {
                tensors.erase(it_out);
                std::cout << "  Removed output tensor " << outputTensor->getGuid() << std::endl;
            }
        }
        // 7. 立即清理弱引用
        std::cout << "Cleaning up weak references immediately after removal" << std::endl;
        cleanupWeakReferences();
        
        std::cout << "=== Transpose pair elimination completed ===" << std::endl;
    }

    void GraphObj::eliminateRedundantTransposes() {
        // 创建算子副本，避免在遍历时修改容器
        OpVec opsCopy = ops;
        
        for (auto &op : opsCopy) {
            // 只处理Transpose算子
            if (op->getOpType() != OpType::Transpose) continue;
            
            auto transposeOp1 = std::dynamic_pointer_cast<TransposeObj>(op);
            if (!transposeOp1) continue;
            
            // 获取当前transpose的输出tensor
            Tensor outputTensor = transposeOp1->getOutput();
            
            // 检查这个输出tensor是否只被一个算子使用
            const auto& targets = outputTensor->getTargets();
            if (targets.size() != 1) continue;
            
            // 获取使用这个输出的下一个算子
            Operator nextOp = targets[0]; // 直接使用shared_ptr，不需要lock()
            if (!nextOp || nextOp->getOpType() != OpType::Transpose) continue;
            
            auto transposeOp2 = std::dynamic_pointer_cast<TransposeObj>(nextOp);
            if (!transposeOp2) continue;
            
            // 检查两个permute是否互为逆置换
            if (isInversePermutation(transposeOp1->getPermute(), 
                                transposeOp2->getPermute())) {
                eliminateTransposePair(transposeOp1, transposeOp2);
            }
        }
    }

    bool GraphObj::isLastTwoDimsSwap(const vector<int>& permute) {
        size_t rank = permute.size();

        if (rank < 2) {
            return false;
        }

        for (size_t i = 0; i < rank - 2; i++) {
            if (permute[i] != static_cast<int>(i)) {
                return false;
            }
        }

        return permute[rank - 2] == static_cast<int>(rank - 1) &&
               permute[rank - 1] == static_cast<int>(rank - 2);
    }

    void GraphObj::fuseTransposeWithMatmul(Ref<MatmulObj> matmulOp, int inputIdx, Ref<TransposeObj> transposeOp){
        std::cout << "\n=== Fusing Transpose with MatMul ===" << std::endl;
        std::cout << "  MatMul: " << matmulOp->getGuid() 
                << ", inputIdx: " << inputIdx
                << ", Transpose: " << transposeOp->getGuid() << std::endl;
        
        // 1. 获取关键tensor
        Tensor transposeInput = transposeOp->getInputs(0);
        Tensor transposeOutput = transposeOp->getOutput();
        Tensor originalInput = matmulOp->getInputs(inputIdx);
        
        std::cout << "  Tensors: transposeInput=" << (transposeInput ? transposeInput->getGuid() : -1)
                << ", transposeOutput=" << (transposeOutput ? transposeOutput->getGuid() : -1)
                << ", originalInput=" << (originalInput ? originalInput->getGuid() : -1) << std::endl;
        
        // 2. 验证所有指针有效性
        if (!transposeInput || !transposeOutput || !originalInput) {
            std::cout << "  ERROR: null tensors detected" << std::endl;
            return;
        }
        
        // 3. 更新MatMul的输入
        // 先移除旧连接
        if (auto oldTarget = std::find(originalInput->getTargets().begin(), originalInput->getTargets().end(), matmulOp);
            oldTarget != originalInput->getTargets().end()) {
            originalInput->removeTarget(matmulOp);
        }
        
        // 添加新连接
        transposeInput->addTarget(matmulOp);
        
        // 替换输入
        matmulOp->inputs[inputIdx] = transposeInput;
        
        std::cout << "  Updated MatMul input: " << originalInput->getGuid() 
                << " -> " << transposeInput->getGuid() << std::endl;
        
        // 4. 更新MatMul的trans属性
        if (inputIdx == 0) {
            bool oldTransA = matmulOp->getTransA();
            matmulOp->setTransA(!oldTransA);
            std::cout << "  Updated transA: " << oldTransA << " -> " << matmulOp->getTransA() << std::endl;
        } else if (inputIdx == 1) {
            bool oldTransB = matmulOp->getTransB();
            matmulOp->setTransB(!oldTransB);
            std::cout << "  Updated transB: " << oldTransB << " -> " << matmulOp->getTransB() << std::endl;
        }
        
        // 5. 从图中移除Transpose算子
        std::cout << "  Removing Transpose operator " << transposeOp->getGuid() << std::endl;
        removeOperator(transposeOp);
        
        // 6. 检查是否可以移除transposeOutput tensor
        // 先清理transposeOutput的所有连接
        transposeOutput->setSource(nullptr);  // 移除source连接
        // 移除所有target连接
        auto targets = transposeOutput->getTargets();
        for (auto& target : targets) {
            if (target) {
                transposeOutput->removeTarget(target);
            }
        }
        
        std::cout << "  transposeOutput targets after cleanup: " << transposeOutput->getTargets().size() << std::endl;
        
        if (transposeOutput->getTargets().empty() && !transposeOutput->getSource()) {
            std::cout << "  Removing transpose output tensor " << transposeOutput->getGuid() << std::endl;
            removeTensor(transposeOutput);
        } else {
            std::cout << "  Keeping transpose output tensor " << transposeOutput->getGuid() 
                    << ": targets=" << transposeOutput->getTargets().size() 
                    << ", source=" << (transposeOutput->getSource() ? "exists" : "none") << std::endl;
        }
        
        // 7. 立即清理弱引用
        std::cout << "  Cleaning up weak references after fusion" << std::endl;
        cleanupWeakReferences();
        
        std::cout << "=== Fusion completed successfully ===\n" << std::endl;
    }

    void GraphObj::fuseTransposeIntoMatmul() {
        // 创建算子副本，避免在遍历时修改容器
        OpVec opsCopy = ops;
        
        for (auto &op : opsCopy) {
            // 只处理Matmul算子
            if (op->getOpType() != OpType::MatMul) continue;
            
            auto matmulOp = std::dynamic_pointer_cast<MatmulObj>(op);
            if (!matmulOp) continue;
            
            // 检查输入A
            Tensor inputA = matmulOp->getInputs(0);
            Operator sourceOpA = inputA->getSource(); // 直接获取source
            if (sourceOpA && sourceOpA->getOpType() == OpType::Transpose) {
                auto transposeOp = std::dynamic_pointer_cast<TransposeObj>(sourceOpA);
                if (transposeOp && isLastTwoDimsSwap(transposeOp->getPermute())) {
                    fuseTransposeWithMatmul(matmulOp, 0, transposeOp);
                }
            }
            
            // 检查输入B
            Tensor inputB = matmulOp->getInputs(1);
            Operator sourceOpB = inputB->getSource();
            if (sourceOpB && sourceOpB->getOpType() == OpType::Transpose) {
                auto transposeOp = std::dynamic_pointer_cast<TransposeObj>(sourceOpB);
                if (transposeOp && isLastTwoDimsSwap(transposeOp->getPermute())) {
                    fuseTransposeWithMatmul(matmulOp, 1, transposeOp);
                }
            }
        }
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        topo_sort();

        // optimize 1:
        eliminateRedundantTransposes();

        // // optimize 2;
        fuseTransposeIntoMatmul();

        topo_sort();
        shape_infer();
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        
        // 第一阶段：只计算偏移量，不实际分配内存
    
        // 1. 为所有输出tensor计算偏移量
        std::vector<std::pair<Tensor, size_t>> outputAllocations;
        for (const auto &op : ops) {
            for (const auto &output : op->getOutputs()) {
                if (output) {
                    size_t size = output->getBytes();
                    if (size > 0) {
                        size_t offset = allocator.alloc(size);
                        outputAllocations.emplace_back(output, offset);
                    }
                }
            }
        }
        
        // 2. 为所有输入tensor计算偏移量
        std::vector<std::pair<Tensor, size_t>> inputAllocations;
        for (const auto &tensor : tensors) {
            // 跳过中间tensor（有source的tensor）
            if (tensor && !tensor->getSource()) {
                size_t size = tensor->getBytes();
                if (size > 0) {
                    size_t offset = allocator.alloc(size);
                    inputAllocations.emplace_back(tensor, offset);
                }
            }
        }
        
        // 第二阶段：实际分配内存
        void* basePtr = allocator.getPtr();
        IT_ASSERT(basePtr != nullptr, "Base pointer is null after allocation");
        
        // 第三阶段：设置tensor的内存指针
        
        // 3. 设置输出tensor的内存
        for (const auto& [tensor, offset] : outputAllocations) {
            void* actualPtr = static_cast<char*>(basePtr) + offset;
            Blob blob = make_ref<BlobObj>(runtime, actualPtr);
            tensor->setDataBlob(blob);
        }
        
        // 4. 设置输入tensor的内存
        for (const auto& [tensor, offset] : inputAllocations) {
            void* actualPtr = static_cast<char*>(basePtr) + offset;
            Blob blob = make_ref<BlobObj>(runtime, actualPtr);
            tensor->setDataBlob(blob);
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini