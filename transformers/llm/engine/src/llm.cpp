//
//  llm.cpp
//
//  Created by MNN on 2023/08/25.
//  ZhaodeWang
//
// #define MNN_OPEN_TIME_TRACE 1

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <unordered_set>

#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include "cpp/ExprDebug.hpp"
#include "llm/llm.hpp"
#include "kvmeta.hpp"
#include "llmconfig.hpp"
#include "prompt.hpp"
#include "tokenizer.hpp"
#include "diskembedding.hpp"
#include "sampler.hpp"
#include "omni.hpp"
#include "speculative_decoding/generate.hpp"

// 0: no debug, 1: test op time, 2: print tensor info, 3: print tensor in output
#define DEBUG_MODE 0
//#define DEBUG_IMAGE

namespace MNN {
using namespace Express;
namespace Transformer {

/**
 * @brief 同步元数据状态，更新previous值并重置相关计数器
 * 
 * 该函数用于同步KVMeta对象的内部状态，通过计算reserve数组中的回滚数量，
 * 结合add和remove计数器来更新previous值，最后重置所有临时计数器。
 * 
 * @param 无参数
 * @return 无返回值
 */
void KVMeta::sync() {
    int revertNumber = 0;
    for (int i=0; i<n_reserve; ++i) {
        revertNumber += reserve[2*i+1];
    }
    previous = previous - remove + add + revertNumber;
    n_reserve = 0;
    reserve = nullptr;
    remove = 0;
    add = 0;
}

/**
 * @brief 将字符串类型的后端类型转换为MNNForwardType枚举值
 * @param type_str 字符串类型的后端类型，支持"cpu"、"metal"、"cuda"、"opencl"、"opengl"、"vulkan"、"npu"等
 * @return 返回对应的MNNForwardType枚举值，如果字符串不匹配则返回MNN_FORWARD_AUTO
 */
static MNNForwardType backend_type_convert(const std::string& type_str) {
    if (type_str == "cpu")
        return MNN_FORWARD_CPU;
    if (type_str == "metal")
        return MNN_FORWARD_METAL;
    if (type_str == "cuda")
        return MNN_FORWARD_CUDA;
    if (type_str == "opencl")
        return MNN_FORWARD_OPENCL;
    if (type_str == "opengl")
        return MNN_FORWARD_OPENGL;
    if (type_str == "vulkan")
        return MNN_FORWARD_VULKAN;
    if (type_str == "npu")
        return MNN_FORWARD_NN;
    return MNN_FORWARD_AUTO;
}

template <typename T>
static inline VARP _var(std::vector<T> vec, const std::vector<int> &dims) {
    return _Const(vec.data(), dims, NHWC, halide_type_of<T>());
}

/**
 * @brief 创建LLM实例的工厂方法
 * @param config_path 配置文件路径
 * @return 返回创建的Llm对象指针，需要调用者负责释放内存
 */
Llm* Llm::createLLM(const std::string& config_path) {
    // 加载配置文件
    std::shared_ptr<LlmConfig> config(new LlmConfig(config_path));
    Llm* llm = nullptr;

    // 根据配置决定创建Omni类型还是普通Llm类型的实例，即多模态还是普通Llm类型
    if (config->is_visual() || config->is_audio() || config->has_talker()) {
        llm = new Omni(config);
    } else {
        llm = new Llm(config);
    }
    return llm;
}

/**
 * @brief 销毁LLM实例
 * @param llm 待销毁的Llm对象指针
 */
void Llm::destroy(Llm* llm) {
    delete llm;
}

/**
 * @brief 将配置信息转换为字符串格式并返回
 * 
 * 该函数调用内部配置对象的dump方法，将配置信息序列化为字符串格式，
 * 通常用于调试或配置信息的输出
 * 
 * @return std::string 返回序列化后的配置信息字符串
 */
std::string Llm::dump_config() {
    return mConfig->config_.dump();
}

/**
 * @brief 设置LLM配置
 * @param content 配置内容字符串
 * @return 配置合并结果，成功返回true，失败返回false
 */
bool Llm::set_config(const std::string& content) {
    auto res = mConfig->config_.merge(content.c_str());
    // 更新提示词配置
    if(mPrompt != nullptr) {
        mPrompt->setParams(mConfig);
    } else {
        mPrompt.reset(Prompt::createPrompt(mContext, mConfig));
    }
    // 设置为异步模式，默认为True
    mAsync = mConfig->config_.document.HasMember("async") ? mConfig->config_.document["async"].GetBool() : true;
    return res;
}

/**
 * @brief 获取模型推理过程中的统计信息，并以JSON格式字符串返回。
 *
 * 该函数收集当前上下文中的各类性能统计数据，包括视觉处理时间、音频处理时间、
 * Prefill阶段耗时、Decode阶段耗时、采样耗时等，并计算Prefill和Decode的速度（tokens/s），
 * 最终将这些数据组织成一个JSON格式的字符串返回。
 *
 * @return std::string 返回包含各项统计信息的JSON格式字符串。
 */
std::string Llm::get_statistics() {
    // 获取当前推理上下文
    auto context = getContext();

    // 提取提示词长度和生成序列长度
    int prompt_len = context->prompt_len;
    int decode_len = context->gen_seq_len;

    // 提取各阶段耗时（单位：微秒）
    int64_t vision_time = context->vision_us;
    int64_t audio_time = context->audio_us;
    int64_t prefill_time = context->prefill_us;
    int64_t decode_time = context->decode_us;
    int64_t sample_time = context->sample_us;

    // 将时间从微秒转换为秒
    float vision_s = vision_time / 1e6;
    float audio_s = audio_time / 1e6;
    float prefill_s = prefill_time / 1e6;
    float decode_s = decode_time / 1e6;
    float sample_s = sample_time / 1e6;

    // 计算Prefill与Decode阶段的处理速度（tokens per second）
    float prefill_speed = (prefill_s > 0.0f) ? (prompt_len / prefill_s) : 0.0f;
    float decode_speed = (decode_s > 0.0f) ? (decode_len / decode_s) : 0.0f;

    // 构造JSON格式的统计信息字符串
    std::ostringstream json_stream;
    json_stream << "{"
                << "\"prompt_tokens\":" << prompt_len << ","
                << "\"decode_tokens\":" << decode_len << ","
                << "\"vision_time\":" << std::fixed << std::setprecision(2) << vision_s << ","
                << "\"audio_time\":" << std::fixed << std::setprecision(2) << audio_s << ","
                << "\"prefill_time\":" << std::fixed << std::setprecision(2) << prefill_s << ","
                << "\"decode_time\":" << std::fixed << std::setprecision(2) << decode_s << ","
                << "\"sample_time\":" << std::fixed << std::setprecision(2) << sample_s << ","
                << "\"prefill_speed\":" << std::fixed << std::setprecision(2) << prefill_speed << ","
                << "\"decode_speed\":" << std::fixed << std::setprecision(2) << decode_speed
                << "}";

    return json_stream.str();
}

/**
 * @brief 设置运行时提示信息，用于配置推理执行器的行为。
 * 
 * 该函数根据配置对象 [mConfig](file:///home/panguofeng/mnt/mnn_deploy/MNN/project/android/demo/app/src/main/java/com/taobao/android/mnndemo/VideoActivity.java#L58-L58) 中的参数，设置运行时管理器 `rtg` 的各种运行时提示，
 * 包括线程数、内存分配器类型、量化选项、KV缓存大小限制、外部路径等。
 * 
 * @param rtg 运行时管理器的智能指针引用，用于设置运行时参数。
 */
void Llm::setRuntimeHint(std::shared_ptr<Express::Executor::RuntimeManager> &rtg) {
    // 设置初始化线程数为4
    rtg->setHint(MNN::Interpreter::INIT_THREAD_NUMBER, 4);
    // 设置内存分配器类型为默认（0）
    rtg->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 0);

    // 设置QKV量化选项，若未配置则默认为8
    rtg->setHint(MNN::Interpreter::QKV_QUANT_OPTIONS, mConfig->config_.value("quant_qkv", 8));

    // 设置KV缓存大小限制
    rtg->setHint(MNN::Interpreter::KVCACHE_SIZE_LIMIT, mConfig->kvcache_limit());

    // 若启用cached mmap，则设置对应标志
    if (mConfig->use_cached_mmap()) {
        rtg->setHint(MNN::Interpreter::USE_CACHED_MMAP, 1);
    }

    // 获取临时路径
    std::string tmpPath = mConfig->tmp_path();

    // 若启用KV缓存mmap，则设置KV缓冲目录
    if (mConfig->kvcache_mmap()) {
        rtg->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_PATH_KVCACHE_DIR);
    }

    // 若启用模型权重mmap,则设置模型权重目录
    if (mConfig->use_mmap()) {
        rtg->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_WEIGHT_DIR);
    }
    
    // 设置NPU模型目录
    rtg->setExternalPath(mConfig->npu_model_dir(), 3);

    // 获取动态量化选项
    auto dynamicOption = mConfig->dynamic_option();

    // 若启用动态量化，则设置对应选项
    if (mConfig->dynamic_option()) {
        rtg->setHint(MNN::Interpreter::DYNAMIC_QUANT_OPTIONS, mConfig->dynamic_option());
    }

    // 根据线程数决定是否启用Arm86内核的SME2指令支持
    if (mConfig->thread_num() > 7) { // if thread_num > 7, cpu dynamic quant use Arm86 kernels
        rtg->setHint(MNN::Interpreter::CPU_SME2_INSTRUCTIONS, 0);
    } else {
        rtg->setHint(MNN::Interpreter::CPU_SME2_INSTRUCTIONS, 1);

    }

    // 若启用prefer_decode模式，则调整动态量化选项
    if (mConfig->config_.value("prefer_decode", false)) {
        dynamicOption = dynamicOption % 8 + 8;
        rtg->setHint(MNN::Interpreter::DYNAMIC_QUANT_OPTIONS, dynamicOption);
    }

    // 设置KV缓存元信息指针
    rtg->setHintPtr(Interpreter::KVCACHE_INFO, mMeta.get());

    // 若后端类型不是CPU，则设置缓存文件路径
    if (backend_type_convert(mConfig->backend_type()) != 0) { // not cpu
        std::string cacheFilePath = tmpPath.length() != 0 ? tmpPath : ".";
        rtg->setCache(cacheFilePath + "/mnn_cachefile.bin");
    }
}


/**
 * @brief 初始化运行时环境，配置后端参数并创建运行时管理器。
 *
 * 该函数根据配置文件中的参数设置运行时的调度配置和后端配置，
 * 包括线程数、功耗模式、内存模式、精度模式等，并根据调试模式
 * 设置相应的调试选项。同时支持 OpenCL 的特殊线程配置。
 *
 * @note 该函数不接受参数，也不返回任何值。
 *       所有配置来源于成员变量 mConfig。
 */
void Llm::initRuntime() {
    ScheduleConfig config;
    BackendConfig cpuBackendConfig;

    // 设置调度类型和线程数
    config.type      = backend_type_convert(mConfig->backend_type());
    config.numThread = mConfig->thread_num();

    // OpenCL 后端需要将线程数设置为 64（缓冲区模式）
    if(config.type == 3){
        config.numThread |= 64;
    }

    // 配置 CPU 后端的功耗模式
    if (mConfig->power() == "high") {
        cpuBackendConfig.power = BackendConfig::Power_High;
    } else if (mConfig->power() == "low") {
        cpuBackendConfig.power = BackendConfig::Power_Low;
    }

    // 配置 CPU 后端的内存模式
    if (mConfig->memory() == "high") {
        cpuBackendConfig.memory = BackendConfig::Memory_High;
    } else if (mConfig->memory() == "low") {
        cpuBackendConfig.memory = BackendConfig::Memory_Low;
    }

    // 配置 CPU 后端的计算精度模式
    if (mConfig->precision() == "high") {
        cpuBackendConfig.precision = BackendConfig::Precision_High;
    } else if (mConfig->precision() == "low") {
        cpuBackendConfig.precision = BackendConfig::Precision_Low;
    }

    // 将后端配置赋值给调度配置
    config.backendConfig = &cpuBackendConfig;

    // 创建并重置运行时管理器
    mRuntimeManager.reset(Executor::RuntimeManager::createRuntimeManager(config));

    // 设置运行时提示信息
    setRuntimeHint(mRuntimeManager);

#if DEBUG_MODE == 1
    // 开启调试模式1：运行时开启时间追踪
    mRuntimeManager->setMode(MNN::Interpreter::Session_Debug);
    _initTimeTrace();
#endif
#if DEBUG_MODE == 2
    // 开启调试模式2：运行时开启静态张量追踪，即启用张量统计
    mRuntimeManager->setMode(MNN::Interpreter::Session_Debug);
    _initTensorStatic();
#endif
#if DEBUG_MODE == 3
    // 开启调试模式3：运行时开启动态张量追踪，即启用通用调试功能
    mRuntimeManager->setMode(MNN::Interpreter::Session_Debug);
    _initDebug();
#endif
    // 如果启用了调试配置，则设置调试模式
    if (mConfig->config_.value("enable_debug", false)) {
        mRuntimeManager->setMode(MNN::Interpreter::Session_Debug);
    }
}

/**
 * @brief 检查模块是否可以进行特殊解码
 * @param module 要检查的模块指针
 * @return 如果模块可以进行特殊解码则返回true，否则返回false
 */
static bool canSpecDecode(std::shared_ptr<Express::Module> module) {
    bool canSpec = false;
    auto info = module->getInfo();
    // 检查MNN模型的输入信息，判断是否包含特定的logits_index输入
    for (int i=0; i<info->inputNames.size(); ++i) {
        auto& varInfo = info->inputs[i];
        if(info->inputNames[i] == "logits_index") {
            if (varInfo.dim.size() > 0) {
                canSpec = true;
            }
        }
    }
    return canSpec;
}

/**
 * @brief 设置特殊解码配置
 * 
 * 该函数根据配置文件中的特殊解码类型设置相关的配置参数。
 * 如果配置了特殊解码且模型支持特定解码，则启用特殊解码模式并设置草稿预测长度。
 * 解码方式："lookahead"、 ”mtp“、 "draftmodel"，多输出解码
 * 
 * @note 该函数无参数和返回值
 */
void Llm::setSpeculativeConfig() {
    auto specultive_type = mConfig->speculative_type();
    if(!specultive_type.empty()) {
        if(!canSpecDecode(mModules[0])) {
            mInSpec = false;
            return;
        }
        mDraftLength = mConfig->draft_predict_length();
        mInSpec = true;
    }
}

/**
 * @brief 加载模型及其相关组件，初始化运行时环境。
 * 
 * 该函数负责完成以下主要任务：
 * 1. 初始化运行时环境；
 * 2. 加载词汇表（Tokenizer）；
 * 3. 初始化嵌入层、提示处理模块和采样器；
 * 4. 根据配置加载主模型，并设置输入输出节点；
 * 5. 根据是否启用推测解码（speculative decoding）克隆多个模型实例；
 * 6. 设置注意力掩码和位置ID等模型输入变量；
 * 7. 加载生成策略及MTP模型（如果启用）。
 */
void Llm::load() {
    initRuntime();
    // init module status
    
    // 1. 加载词汇表（Tokenizer）
    mTokenizer.reset(Tokenizer::createTokenizer(mConfig->tokenizer_file())); // tokenizer init 读取词汇表
    
    // 初始化磁盘嵌入模块
    mDiskEmbedding.reset(new DiskEmbedding(mConfig)); //知识库嵌入
    
    // 2. 初始化Prompt处理模块
    mPrompt.reset(Prompt::createPrompt(mContext, mConfig)); //提示词初始化
    
    // 初始化采样器模块
    mSampler.reset(Sampler::createSampler(mContext, mConfig)); //采样器初始化
    
    // 3. 根据配置加载主模型
    Module::Config module_config;
    if (mConfig->backend_type() == "opencl" || mConfig->backend_type() == "vulkan") {
        module_config.shapeMutable = false;
    } else {
        module_config.shapeMutable = true;
    }
    module_config.rearrange    = true;
    
    // 若存在基础模块，则使用LoRA模块
    if (mBaseModule != nullptr) {
        module_config.base = mBaseModule;
    }

    // 加载单个模型
    mModules.resize(1);
    std::string model_path = mConfig->llm_model();

    // 定义模型的输入输出名称
    std::vector<std::string> inputNames {"input_ids", "attention_mask", "position_ids", "logits_index"};
    std::vector<std::string> outputNames {"logits"};

    // 如果启用了talker功能，则添加额外输出
    if (mConfig->has_talker()) {
        outputNames.emplace_back("talker_embeds");
    }

    // 检查是否需要输出隐藏hidden_states状态
    bool needHiddenState = false;
    if (mConfig->config_.document.HasMember("hidden_states")) {
        needHiddenState = mConfig->config_.document["hidden_states"].GetBool();
    }
    if(mConfig->speculative_type() == "mtp") {
        needHiddenState = true;
    }
    if (needHiddenState) {
        outputNames.emplace_back("hidden_states");
    }

    // 设置外部权重文件路径并加载模型
    mRuntimeManager->setExternalFile(mConfig->llm_weight());
    mModules[0].reset(Module::load(inputNames, outputNames, model_path.c_str(), mRuntimeManager, &module_config));
    mRuntimeManager->setExternalFile("");
    
    // 检查模型加载是否成功
    if(nullptr == mModules[0]) {
        MNN_ERROR("[Error]: Load module failed, please check model.\n");
        if(outputNames.size() > 1) {
            MNN_ERROR("[Warning]: Set module multi outputs, please double check.\n");
        }
        return;
    }

    // 设置推测解码相关参数
    setSpeculativeConfig();
    
    // 创建生成策略实例
    mGenerationStrategy = GenerationStrategyFactory::create(this, mContext, mConfig, mInSpec);

    int decode_type_num = 1;
    int verify_length = 1;

    // 如果启用推测解码，则克隆验证阶段使用的模型实例
    if(mInSpec) {
        // decode one token or mDraftLength token
        decode_type_num = 2;
        verify_length = mDraftLength + 1;
        // speculative decode module
        mModulePool[std::make_pair(verify_length, true)].reset(Module::clone(mModules[0].get()));
    }

    // 克隆自回归解码使用的模型实例
    mModulePool[std::make_pair(1, false)].reset(Module::clone(mModules[0].get()));
    
    // 将prefill阶段使用的模型加入池中
    mModulePool[std::make_pair(mPrefillKey, mConfig->all_logits())] = mModules[0];

    // 初始化logits索引变量
    logitsLastIdx = _var<int>({-1}, {1});
    logitsAllIdx = _var<int>({0}, {1});
    
    // 初始化注意力掩码和位置ID变量向量
    mAttentionMaskVarVec.resize(decode_type_num);
    mPositionIdsVarVec.resize(decode_type_num);
    for(int i = 0; i < decode_type_num; i++) {
        int index = 1;
        if(i > 0) {
            index = verify_length;
        }
        
        // 构造注意力掩码变量
        {
            mAttentionMaskVarVec[i] = _Input({1, 1, index, index}, NCHW, halide_type_of<float>());
            auto ptr = mAttentionMaskVarVec[i]->writeMap<float>();
            for (int i = 0; i < index; i++) {
                for (int j = 0; j < index; j++) {
                    ptr[index * i + j] = (j > i) * std::numeric_limits<float>::lowest();
                }
            }
        }
        
        // 构造位置ID变量
        mPositionIdsVarVec[i] = _Input({index}, NCHW, halide_type_of<int>());
    }

    // MTP model load
    // 加载生成策略所需的模型配置（MTP），多Token预测推理模型，加速推理效率 - 推理阶段加速方法
    mGenerationStrategy->load(module_config);
}


/**
 * @brief 创建一个LoRA微调模型实例
 * 
 * 该函数通过复制当前模型的配置，创建一个新的LoRA模型实例，
 * 并加载指定路径的LoRA权重文件
 * 
 * @param lora_path LoRA权重文件的路径
 * @return 返回新创建的LoRA模型指针
 */
Llm* Llm::create_lora(const std::string& lora_path) {
    // 创建新的LLM实例，复制当前配置
    auto llm = new Llm(std::make_shared<LlmConfig>(*mConfig));

    // 设置模型配置，指定LoRA模型路径并禁用内存映射
    llm->set_config("{\"llm_model\": \"" + lora_path + "\", \"use_mmap\": false, \"use_cached_mmap\": false}");
    
    // 设置基础模块为当前模块列表的第一个模块
    llm->mBaseModule = mModules.begin()->get();
    
    // 加载模型权重
    llm->load();
    return llm;
}

/**
 * @brief 对模型进行调优，以选择最优的编码器数量配置。
 * 
 * 该函数根据给定的候选编码器数量，通过实际运行推理来评估不同配置下的性能，
 * 并选择耗时最短的配置作为最终设置。主要用于优化解码阶段的性能。
 * 
 * @param type 调优类型，当前仅支持 OP_ENCODER_NUMBER 类型。
 * @param candidates 候选的编码器数量列表，用于测试不同配置的性能。
 */
void Llm::tuning(TuneType type, std::vector<int> candidates) {
    // 检查调优类型是否为支持的 OP_ENCODER_NUMBER
    if (type != OP_ENCODER_NUMBER) {
        MNN_ERROR("tuning type not supported\n");
        return;
    }

    // FIXME: 当前 OpenCL 后端不支持 KVMeta，因此跳过调优
    if (mConfig->backend_type() == "opencl") {
        return;
    }

    int decode_seq = 1;

    // 设置为解码模式，生成序列长度设为 1
    mContext->gen_seq_len = 1;

    // 如果存在输入规范，则启动自回归解码流程
    if(mInSpec) {
        // 构造初始输入并执行一次前向推理
        std::vector<int> input_ids = {0};
        auto logits = forwardVec(input_ids);

        // 计算验证长度，用于后续解码序列长度设置
        int verify_length = mDraftLength + 1;
        decode_seq = verify_length;
    }

    int64_t min_time     = INT64_MAX;
    int prefer_candidate = 10;

    // 遍历所有候选编码器数量，测试其推理耗时
    for (auto& candidate : candidates) {
        // 设置当前候选编码器数量
        mRuntimeManager->setHint(MNN::Interpreter::OP_ENCODER_NUMBER_FOR_COMMIT, candidate);

        // 启动计时器并构造输入数据
        Timer _t;
        std::vector<int> input_ids(decode_seq, 0);
        auto outputs = forwardVec(input_ids);

        // 检查输出是否有效
        if(outputs.empty()) {
            return;
        }
        auto logits = outputs[0];
        if (nullptr == logits.get()) {
            return;
        }
        if (logits->getInfo()->size == 0) {
            return;
        }

        // 采样获取 token，并记录耗时
        auto token = sample(logits);
        auto time = _t.durationInUs();

        // 更新最优候选配置
        if (time < min_time) {
            prefer_candidate = candidate;
            min_time         = time;
            // MNN_PRINT("op encode number:%d, decode time: %lld us\n", candidate, time);
        }
    }

    // 应用最优的编码器数量配置
    mRuntimeManager->setHint(MNN::Interpreter::OP_ENCODER_NUMBER_FOR_COMMIT, prefer_candidate);

    // 清理调优过程中产生的脏 KV 缓存历史
    setKVCacheInfo(0, getCurrentHistory());
    reset();
}

void Llm::switchMode(Llm::Stage stage) {
    // do nothing, only reserve api
    return;
}

/**
 * @brief 设置KV缓存信息
 * 
 * @param add 需要添加的缓存数量
 * @param remove 需要移除的缓存数量
 * @param reserve 预留缓存数组指针
 * @param n_reserve 预留缓存数组长度
 */
void Llm::setKVCacheInfo(size_t add, size_t remove, int* reserve, int n_reserve) {
    // 确保移除数量不超过之前的缓存数量
    if (remove > mMeta->previous) {
        remove = mMeta->previous;
    }

    mMeta->remove = remove;
    mMeta->reserve = reserve;
    mMeta->n_reserve = n_reserve;
    mMeta->add = add;
}


/**
 * @brief 执行模型前向推理的核心函数，根据当前状态和配置选择合适的模块进行推理。
 *
 * 此函数会判断当前是处于预填充（prefill）阶段还是解码（decode）阶段，并据此决定是否需要克隆新模块，
 * 同时控制输出的是全部logits还是仅最后一个位置的logits。此外，在调试模式下还会打印中间张量统计信息。
 *
 * @param hiddenState 输入的隐藏状态，形状通常为 [batch_size, seq_len, hidden_dim]
 * @param mask 注意力掩码，用于屏蔽无效位置，形状一般为 [batch_size, 1, 1, seq_len]
 * @param inputPos 当前输入token在序列中的位置索引，形状为 [batch_size]
 * @return 返回模型推理结果，通常是logits或其他输出变量组成的向量
 */
std::vector<Express::VARP> Llm::forwardRaw(Express::VARP hiddenState, Express::VARP mask, Express::VARP inputPos) {
    Express::VARP logitsIndex;
    // 判断当前是否处于解码阶段
    bool inDecode = mContext->gen_seq_len > 0;
    // 根据配置及当前阶段确定是否输出所有logits
    bool isAllLogists = mConfig->all_logits() ? true : (inDecode ? mInSpec : false);
    // 获取当前key长度：若为解码则取hiddenState第一维，否则使用预填充key
    int seqLenKey = inDecode ? hiddenState->getInfo()->dim[0] : mPrefillKey;
    // 若序列长度为1，则强制不输出全部logits
    isAllLogists = seqLenKey == 1 ? false : isAllLogists;
    // 构造模块池键值对，用于查找或创建对应模块
    auto moduleKey = std::make_pair(seqLenKey, isAllLogists);

    // 检查是否有对应的已缓存模块，如果没有则克隆一个新的
    if(mModulePool.find(moduleKey) == mModulePool.end()) {
        MNN_PRINT("Warning: module need new clone, cloning now.\n");
        // 设置运行时提示指针以传递KV Cache相关信息
        mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mMeta.get());
        // 克隆主模块并保存到模块池中
        mModulePool[moduleKey].reset(Module::clone(mModules[0].get()));
    }

    // 根据是否输出全部logits设置相应的索引变量
    if (isAllLogists) {
        logitsIndex = logitsAllIdx;
    } else {
        logitsIndex = logitsLastIdx;
    }

    // 清空生成参数中的旧数据
    mGenerateParam->input_embeds = nullptr;
    mGenerateParam->outputs.clear();

    // 调用选中的模块执行前向计算
    std::vector<Express::VARP> outputs;
    outputs = mModulePool[moduleKey]->onForward({hiddenState, mask, inputPos, logitsIndex});

    // 如果没有输出直接返回空结果
    if (outputs.empty()) {
        return outputs;
    }

    // 非异步模式下等待tensor完成读取操作
    if (!mAsync) {
        ((MNN::Tensor*)(outputs[0]->getTensor()))->wait(Tensor::MAP_TENSOR_READ, true);
    }

    // 更新生成参数中的输入与输出
    mGenerateParam->input_embeds = hiddenState;
    mGenerateParam->outputs = outputs;

#if DEBUG_MODE == 3
    // 在调试模式下打印部分张量的统计信息
    VARP logits = outputs[0];
    if(logits->getInfo()->dim[1] < 10 && logits->getInfo()->dim[1] >= 1) {
        for (int j = 0; j < logits->getInfo()->dim[1]; j++) {
            {
                // 统计hiddenState的部分维度数据
                int length = hiddenState->getInfo()->dim[2];
                float total = 0.0;
                float max_ = std::numeric_limits<float>::lowest();
                float min_ = std::numeric_limits<float>::max();
                for (int i = 0; i < length; i++) {
                    int index = j * length + i;
                    float temp = hiddenState->readMap<float>()[index];
                    total += temp;
                    max_ = fmax(max_, temp);
                    min_ = fmin(min_, temp);
                }
                MNN_PRINT("\nhiddenState statistic value:%6f, %6f, %6f\n", total, max_, min_);
            }

            {
                // 统计mask的部分维度数据
                int length = mask->getInfo()->dim[3];
                float total = 0.0;
                float max_ = std::numeric_limits<float>::lowest();
                float min_ = std::numeric_limits<float>::max();
                for (int i = 0; i < length; i++) {
                    int index = j * length + i;
                    float temp = mask->readMap<float>()[index];
                    total += (temp / length);
                    max_ = fmax(max_, temp);
                    min_ = fmin(min_, temp);
                }
                MNN_PRINT("mask statistic value:%6f, %6f, %6f\n", total, max_, min_);
            }

            // 打印当前位置信息
            MNN_PRINT("position statistic value:%d\n", inputPos->readMap<int>()[j]);
            {
                int length = logits->getInfo()->dim[2];
                float total = 0.0;
                float max_ = std::numeric_limits<float>::lowest();
                float min_ = std::numeric_limits<float>::max();
                for (int i = 0; i < length; i++) {
                    int index = j * length + i;
                    float temp = logits->readMap<float>()[index];
                    total += temp;
                    max_ = fmax(max_, temp);
                    min_ = fmin(min_, temp);
                }
                auto ptr = logits->readMap<float>() + j * logits->getInfo()->dim[2];
                //            MNN_PRINT("\noutput data value:%6f %6f %6f %6f %6f\n", ptr[0], ptr[length/5], ptr[length/10], ptr[length/20], ptr[length/100]);
                MNN_PRINT("output statistic value:%6f, %6f, %6f\n", total, max_, min_);
            }
        }
    }
#endif
    // 同步元数据
    mMeta->sync();
    return outputs;
}

/**
 * @brief 执行模型前向传播
 * 
 * 该函数接收输入的token ID序列，首先通过embedding层转换为隐藏状态，
 * 然后执行后续的前向传播计算。
 * 
 * @param input_ids 输入的token ID序列
 * @param is_prefill 是否为预填充阶段（该参数在当前实现中未使用）
 * @return 前向传播结果的张量
 */
VARP Llm::forward(const std::vector<int>& input_ids, bool is_prefill) {
    // 获取输入序列的嵌入表示
    auto hidden_states = embedding(input_ids);
    // 执行后续层的前向传播
    return forward(hidden_states);
}

/**
 * @brief 执行模型前向推理过程
 * @param input_embeds 输入的嵌入向量，包含序列数据
 * @return 返回推理结果的 logits，如果推理失败则返回 nullptr
 */
VARP Llm::forward(MNN::Express::VARP input_embeds) {    
    // 获取输入序列的长度
    int seq_len         = input_embeds->getInfo()->dim[mSeqLenIndex];
    
    // 执行前向推理，获取输出向量
    auto out = forwardVec(input_embeds);

    // 检查推理结果是否为空
    if (out.empty()) {
        return nullptr;
    }

    // 获取第一个输出作为 logits
    auto logits = out[0];

    // 更新上下文状态，传入序列长度和批次大小
    updateContext(seq_len, 1);
    return logits;
}

/**
 * @brief 执行LLM模型的前向传播，输入为token ID序列
 * 
 * 该函数将输入的token ID序列转换为嵌入向量，然后执行前向传播计算，
 * 返回模型的输出结果。
 * 
 * @param input_ids 输入的token ID序列，每个ID对应词汇表中的一个词
 * @return std::vector<VARP> 模型前向传播的输出结果，包含各个层的输出张量
 */
std::vector<VARP> Llm::forwardVec(const std::vector<int>& input_ids) {
    auto input_embeds = embedding(input_ids);
    auto outputs = forwardVec(input_embeds);
    return outputs;
}

/**
 * @brief 执行LLM模型的前向推理过程
 * @param input_embeds 输入的嵌入向量，包含序列的词嵌入表示
 * @return 返回模型推理结果的向量，包含每个时间步的输出logits
 * 
 * 该函数完成以下主要步骤：
 * 1. 从输入嵌入中提取序列长度信息
 * 2. 更新模型元数据中的序列长度
 * 3. 生成注意力掩码和位置ID
 * 4. 调用底层前向推理函数获得最终输出
 */
std::vector<VARP> Llm::forwardVec(MNN::Express::VARP input_embeds) {
    // 获取输入序列的长度
    int seq_len         = input_embeds->getInfo()->dim[mSeqLenIndex];
    // 更新模型元数据中的序列长度信息
    mMeta->add         = seq_len;
    // 生成注意力掩码，用于屏蔽无效位置的注意力计算
    auto attention_mask = gen_attention_mask(seq_len);
    // 生成位置ID，用于表示序列中每个位置的编码
    auto position_ids = gen_position_ids(seq_len);
    // 执行实际的前向推理过程
    auto logits = forwardRaw(input_embeds, attention_mask, position_ids);
    return logits;
}

/**
 * @brief 更新上下文中的序列长度信息
 * 
 * 该函数用于更新模型上下文中的总序列长度和生成序列长度统计信息。
 * 
 * @param seq_len 当前输入序列的长度
 * @param gen_len 当前生成序列的长度
 */
void Llm::updateContext(int seq_len, int gen_len) {
    mContext->all_seq_len += seq_len;
    mContext->gen_seq_len += gen_len;
}

/**
 * 从给定的logits中采样得到token ID
 * @param logits 包含词汇表上各个token概率的张量
 * @param offset 采样起始位置的偏移量，为0时表示从头开始
 * @param size 采样范围的大小，为0时表示使用全部数据
 * @return 采样得到的token ID
 */
int Llm::sample(VARP logits, int offset, int size) {
    // 获取logits的维度信息
    auto logitsShape = logits->getInfo()->dim;

    // 如果指定了偏移量和大小，则截取对应的logits片段
    if (offset && size) {
        logits = _Const(logits->readMap<float>() + offset, {size}, NHWC, halide_type_of<float>());
    }

    // 使用采样器进行采样得到token ID
    auto token_id = mSampler->sample(logits);
    return token_id;
}

/**
 * @brief 重置LLM模型的状态
 * 
 * 该函数用于清除模型的上下文状态，包括输出tokens、历史tokens和序列长度all_seq_len和gen_seq_len等信息,
 * 并恢复元数据到之前的状态。
 * 
 * @note 此函数不接受任何参数，无返回值
 */
void Llm::reset() {
    // 清除输出令牌和历史令牌，重置序列长度计数器
    mContext->output_tokens.clear();
    mContext->history_tokens.clear();
    mContext->all_seq_len = 0;
    mContext->gen_seq_len = 0;

    // 恢复元数据到前一个状态
    mMeta->remove = mMeta->previous;
}

/**
 * @brief 初始化文本生成过程的状态
 * 
 * 该函数用于重置和初始化LLM文本生成相关的各种状态变量，
 * 为下一次文本生成做好准备。
 * 
 * @param os 输出流指针，用于输出生成的文本
 * @param end_with 结束标记字符串，当生成的文本包含此标记时停止生成
 */
void Llm::generate_init(std::ostream* os, const char* end_with) {
    // 初始化基本状态
    mContext->os = os;
    if (nullptr != end_with) {
        mContext->end_with = end_with;
    }

    // 清空之前生成的文本并重置相关计数器
    if (!mContext->generate_str.empty()) {
        mContext->generate_str.clear();
    }
    mContext->gen_seq_len = 0;
    mContext->prefill_us  = 0;
    mContext->decode_us   = 0;
    mContext->current_token = -1;
    mContext->sample_us = 0;

    // 根据配置决定是否重用KV缓存，如果不重用则清空历史记录
    if (!mConfig->reuse_kv()) {
        mContext->all_seq_len = 0;
        mContext->history_tokens.clear();
        mMeta->remove = mMeta->previous;
    }

    // 清空输出token列表
    mContext->output_tokens.clear();
}

/**
 * @brief 获取当前历史记录的位置
 * 
 * @return size_t 返回当前历史记录的位置索引
 */
size_t Llm::getCurrentHistory() const {
    return mMeta->previous;
}

/**
 * @brief 从历史记录中删除指定范围的内容
 * @param begin 要删除的历史记录起始位置
 * @param end 要删除的历史记录结束位置，如果为0则表示删除到最新位置
 * 
 * 该函数用于删除LLM模型中的历史对话记录。可以通过指定begin和end参数来确定
 * 删除的范围。删除操作不会立即执行，而是设置相应的标记位，等待后续响应时处理。
 */
void Llm::eraseHistory(size_t begin, size_t end) {
    // 如果end为0，则将end设置为当前历史记录的前一个位置
    if (0 == end) {
        end = mMeta->previous;
    }

    // 检查删除范围的有效性
    if (end > mMeta->previous || begin >= end) {
        MNN_ERROR("Invalid erase range history larger than current\n");
        return;
    }

    // 检查是否有未完成的删除操作
    if (mMeta->remove != 0) {
        MNN_ERROR("MNN-LLM: erase history hasn't been executed by response, override erase info\n");
    }

    // 设置要删除的历史记录数量
    mMeta->remove = mMeta->previous - begin;

    // 如果删除范围不等于当前全部历史记录，则需要保留部分数据
    if (end != mMeta->previous) {
        mMeta->reserveHost.resize(2);
        mMeta->reserve = mMeta->reserveHost.data();
        mMeta->n_reserve = 1;
        mMeta->reserve[0] = end - begin;
        mMeta->reserve[1] = mMeta->previous - end;
    }
}

/**
 * @brief 检查模型是否应该停止生成
 * 
 * 该函数通过判断当前token是否为停止token来确定模型是否应该停止生成文本。
 * 
 * @return bool 返回true表示模型应该停止生成，false表示继续生成
 */
bool Llm::stoped() {
    return is_stop(mContext->current_token);
}

/**
 * @brief 生成指定数量的新token
 * @param max_token 最大新生成token数量
 * 
 * 该函数用于控制LLM模型的文本生成过程。首先检查当前token是否为停止符，
 * 如果是则直接返回；否则设置最大新生成token数量，并调用生成策略执行实际生成。
 */
void Llm::generate(int max_token) {
    // 检查当前token是否为停止符，如果是则终止生成
    if (is_stop(mContext->current_token)) {
        return;
    }
    // 设置生成参数并执行生成策略
    mGenerateParam->max_new_tokens = max_token;
    mGenerationStrategy->generate(*mGenerateParam);
}

/**
 * @brief 根据输入的token ID序列生成新的token序列
 * 
 * 该函数是LLM模型的文本生成接口，接收输入token ID序列并生成指定数量的新token
 * 
 * @param input_ids 输入的token ID序列，作为生成的起始上下文
 * @param max_tokens 最大生成token数量，如果为负数则使用配置中的默认值
 * @return 生成的token ID序列
 */
std::vector<int> Llm::generate(const std::vector<int>& input_ids, int max_tokens) {
    // 处理最大生成token数量参数，负数时使用默认配置值
    if (max_tokens < 0) {
        max_tokens = mConfig->max_new_tokens();
    }

    // 更新上下文中的提示长度和历史token记录
    mContext->prompt_len = static_cast<int>(input_ids.size());
    mContext->history_tokens.insert(mContext->history_tokens.end(), input_ids.begin(), input_ids.end()); // push to history_ids_
    
    // 执行embedding层处理，将token ID转换为隐藏状态向量
    auto hidden_states = embedding(input_ids);

    // 调用重载版本的generate函数进行实际的文本生成
    return generate(hidden_states, max_tokens);
}

/**
 * @brief 对用户输入内容进行分词编码
 * 
 * 该函数使用内部的分词器对用户提供的文本内容进行编码，
 * 将文本转换为模型可处理的整数序列。
 * 
 * @param user_content 用户输入的文本内容
 * @return std::vector<int> 编码后的整数序列
 */
std::vector<int> Llm::tokenizer_encode(const std::string& user_content) {
    return mTokenizer->encode(user_content);
}

/**
 * @brief 对多模态提示进行分词编码
 * 
 * 该函数使用内部的分词器对多模态输入中的文本模板进行编码，
 * 将文本转换为对应的token序列
 * 
 * @param multimodal_input 包含待编码文本模板的多模态提示对象
 * @return std::vector<int> 编码后的token ID序列
 */
std::vector<int> Llm::tokenizer_encode(const MultimodalPrompt& multimodal_input) {
    return mTokenizer->encode(multimodal_input.prompt_template);
}

/**
 * @brief 处理多模态输入并生成响应
 * 
 * 该函数接收多模态提示输入，对其进行预处理（如模板应用、编码等），
 * 然后调用底层的响应生成函数来生成输出。
 * 
 * @param multimodal_input 包含多模态数据的输入提示，可能包含文本、图像、音频等
 * @param os 输出流指针，用于输出生成的响应内容
 * @param end_with 结束标记字符串，当生成的内容包含此字符串时停止生成
 * @param max_new_tokens 最大新生成的token数量限制
 */
void Llm::response(const MultimodalPrompt& multimodal_input, 
                   std::ostream* os, const char* end_with, int max_new_tokens) {
    //获取提示模板
    auto prompt = multimodal_input.prompt_template;

    // 如果配置要求使用模板，则应用模板处理
    if (mConfig->use_template()) {
        prompt = mPrompt->applyTemplate(prompt, true);
    }
    
    //初始化各种统计时间变量
    int prompt_len = 0;
    int decode_len = 0;
    int64_t vision_time = 0;
    int64_t audio_time = 0;
    int64_t prefill_time = 0;
    int64_t decode_time = 0;
    int64_t sample_time = 0;
    
    // 对多模态输入进行token编码处理
    std::vector<int> input_ids = tokenizer_encode(multimodal_input);

    // 调用重载的response函数进行实际的响应生成
    response(input_ids, os, end_with, max_new_tokens);
}

/**
 * @brief 根据输入的嵌入向量生成指定长度的文本序列
 * @param input_embeds 输入的嵌入向量，包含提示词信息
 * @param max_tokens 最大生成token数量，如果为负数则使用配置中的默认值
 * @return 生成的token序列
 */
std::vector<int> Llm::generate(MNN::Express::VARP input_embeds, int max_tokens) {
    // 处理最大生成长度参数，负数时使用默认配置
    if (max_tokens < 0) {
        max_tokens = mConfig->max_new_tokens();
    }

    // 获取输入序列长度并更新上下文
    int seqLen = input_embeds->getInfo()->dim[mSeqLenIndex];
    mContext->prompt_len = seqLen;

    // 执行前向传播计算
    Timer _t;
    forwardVec(input_embeds);

    // 检查输出结果有效性
    if(mGenerateParam->outputs.size() < 1) {
        return {};
    }

    // 更新上下文状态
    updateContext(seqLen, 0);
    mContext->prefill_us = _t.durationInUs();

    // 前向传播完成后进行垃圾回收
    MNN::Express::ExecutorScope::Current()->gc(); // after prefill

#if DEBUG_MODE == 3
    // 调试模式下保存输入嵌入和logits到文件
    {
        std::ofstream outFile("input_embeds.txt");
        auto temp = input_embeds->readMap<float>();
        for (size_t i = 0; i < input_embeds->getInfo()->size; ++i) {
            outFile << temp[i] << " "; // 每个数字后加空格
        }
        outFile.close();
    }
    {
        std::ofstream outFile("logits.txt");
        auto temp = mGenerateParam->outputs[0]->readMap<float>();
        for (size_t i = 0; i < mGenerateParam->outputs[0]->getInfo()->size; ++i) {
            outFile << temp[i] << " "; // 每个数字后加空格
        }
        outFile.close();
    }
#endif

    // 重置计时器并执行文本生成
    _t.reset();
    // call generation function
    mGenerateParam->max_new_tokens = max_tokens;
    mGenerationStrategy->generate(*mGenerateParam);
    return mContext->output_tokens;
}


/**
 * @brief 生成LLM响应
 * 
 * 该函数接收输入token ID序列，生成相应的文本响应并输出到指定流中。
 * 
 * @param input_ids 输入的token ID序列
 * @param os 输出流指针，用于输出生成的文本
 * @param end_with 结束标记字符串，默认为换行符
 * @param max_new_tokens 最大新生成token数量
 */
void Llm::response(const std::vector<int>& input_ids, std::ostream* os, const char* end_with, int max_new_tokens) {
    // 如果未指定结束标记，则使用默认的换行符
    if (!end_with) { end_with = "\n"; }

    // 初始化生成器
    generate_init(os, end_with);

    // 执行文本生成
    generate(input_ids, max_new_tokens);
}

/**
 * @brief 生成LLM响应
 * @param input_embeds 输入嵌入向量
 * @param os 输出流指针，用于输出生成的文本
 * @param end_with 结束标记字符串，默认为换行符
 * @param max_new_tokens 最大新生成token数量
 */
void Llm::response(MNN::Express::VARP input_embeds, std::ostream* os, const char* end_with, int max_new_tokens) {
    // 设置默认结束标记
    if (!end_with) { end_with = "\n"; }

    // 初始化生成器
    generate_init(os, end_with);

    // 执行文本生成
    generate(input_embeds, max_new_tokens);
}

/**
 * @brief 处理用户输入并生成响应
 * 
 * 该函数接收用户输入内容，根据配置决定是否使用模板处理，
 * 然后将处理后的提示词进行分词编码，最后调用底层响应函数生成输出。
 * 
 * @param user_content 用户输入的内容字符串
 * @param os 输出流指针，用于输出生成的响应内容
 * @param end_with 结束标记字符串，当生成内容包含此标记时停止生成
 * @param max_new_tokens 最大新生成的token数量限制
 */
void Llm::response(const std::string& user_content, std::ostream* os, const char* end_with, int max_new_tokens) {
    // 获取用户输入作为初始提示词
    auto prompt = user_content;

    // 如果配置要求使用模板，则应用模板处理用户输入
    if (mConfig->use_template()) {
        prompt = mPrompt->applyTemplate(user_content, true);
    }

    // 对处理后的提示词进行分词编码
    std::vector<int> input_ids = tokenizer_encode(prompt);

    // 调用重载的response函数进行实际的响应生成
    response(input_ids, os, end_with, max_new_tokens);
}

/**
 * @brief 根据聊天提示生成响应
 * 
 * 该函数接收聊天消息列表，将其转换为模型输入格式，并调用底层响应函数生成回复。
 * 如果聊天提示为空，则直接返回。
 * 
 * @param chat_prompts 聊天消息列表，包含对话历史和当前提问
 * @param os 输出流指针，用于输出生成的响应内容
 * @param end_with 结束标记字符串，当生成内容包含此标记时停止生成
 * @param max_new_tokens 最大新生成token数量限制
 */
void Llm::response(const ChatMessages& chat_prompts, std::ostream* os, const char* end_with, int max_new_tokens) {
    if (chat_prompts.empty()) {
        return;
    }
    auto prompt = mPrompt->applyTemplate(chat_prompts);
    std::vector<int> input_ids = tokenizer_encode(prompt);
    response(input_ids, os, end_with, max_new_tokens);
}

/**
 * @brief Llm类的构造函数，用于初始化大语言模型实例
 * @param config 指向LlmConfig配置对象的智能指针，包含模型配置信息
 */
Llm::Llm(std::shared_ptr<LlmConfig> config) : mConfig(config) {
    // 初始化模型上下文环境
    mContext.reset(new LlmContext);

    // 初始化KV缓存元数据
    mMeta.reset(new KVMeta);

    // 初始化文本生成参数
    mGenerateParam.reset(new GenerationParams);
}

/**
 * @brief Llm析构函数，负责清理资源并输出调试信息
 * 
 * 该函数在DEBUG_MODE为1时会统计并打印各操作类型的执行时间、FLOPS等性能数据，
 * 然后依次释放生成参数、模块列表、运行时管理器等资源。
 */
Llm::~Llm() {
#if DEBUG_MODE == 1
    // 如果启用了时间追踪且存在追踪信息，则计算并打印性能统计数据
    if (nullptr != gTimeTraceInfo) {
        float opSummer       = 0.0f;            // 所有操作的总时间
        float opFlopsSummber = 0.0f;            // 所有操作的总FLOPS

        // 遍历所有操作类型，累加各自的执行时间和FLOPS
        for (auto& iter : gTimeTraceInfo->mTypes) {
            float summer      = 0.0f;           // 当前操作类型的总时间
            float summerflops = 0.0f;           // 当前操作类型的总FLOPS

            // 遍历当前操作类型下的所有时间记录
            for (auto& t : iter.second) {
                for (auto& t0 : t.second) {
                    summer += t0.first;          // 累加执行时间
                    summerflops += t0.second;    // 累加FLOPS       
                }
            }
            // summer      = summer;
            // summerflops = summerflops;
            // 打印当前操作类型的统计信息（总时间、总FLOPS、GFLOPS速度）
            MNN_PRINT("%s : %.7f, FLOP: %.7f, Speed: %.7f GFlops\n", iter.first.c_str(), summer, summerflops,
                      summerflops / summer);
            opSummer += summer;                     // 累加到全局总时间
            opFlopsSummber += summerflops;          // 累加到全局总FLOPS
        }
        MNN_PRINT("OP Summer: %.7f, Flops: %.7f, Speed: %.7f GFlops\n", opSummer, opFlopsSummber,
                  opFlopsSummber / opSummer);
    }
#endif

    // 重置生成参数智能指针，释放相关资源
    mGenerateParam.reset();

    // 重置模块列表和运行时管理器，释放相关资源
    mModules.clear();
    mRuntimeManager.reset();

    // 重置处理运行时管理器智能指针
    mProcessorRuntimeManager.reset();
}

/**
 * @brief 根据输出名称获取其在输出列表中的索引位置
 * 
 * @param name 要查找的输出名称
 * @return int 输出名称对应的索引，如果未找到或模块池为空则返回-1
 */
int Llm::getOutputIndex(const std::string& name) const {
    // 检查模块池是否为空
    if (mModulePool.empty()) {
        return -1;
    }

    // 获取第一个模块的信息
    auto info = mModulePool.begin()->second->getInfo();

    // 遍历模块输出名称列表，查找匹配的名称
    for (int i=0; i<info->outputNames.size(); ++i) {
        if (info->outputNames[i] == name) {
            return i;
        }
    }
    return -1; // 未找到匹配的输出名称
}


/**
 * @brief 获取模型的输出变量列表
 * 
 * @return std::vector<Express::VARP> 包含模型所有输出变量的向量
 * 
 * 该函数返回模型生成参数中存储的输出变量列表，
 * 这些变量代表了模型前向计算的最终输出结果。
 */
std::vector<Express::VARP> Llm::getOutputs() const {
    return mGenerateParam->outputs;
}

/**
 * @brief 检查是否启用KV缓存重用功能
 * 
 * 该函数用于查询当前配置中是否启用了KV缓存重用功能。
 * KV缓存重用是一种优化技术，可以在推理过程中重用先前计算的键值对，
 * 从而减少重复计算，提高推理效率。
 * 
 * @return bool 返回true表示启用KV缓存重用，false表示禁用
 */
bool Llm::reuse_kv() { return mConfig->reuse_kv(); }

/**
 * 判断是否需要创建新的变量
 * 
 * @param var 需要检查的变量指针
 * @param axis 用于比较的维度索引
 * @param seq_len 期望的序列长度
 * @param kv_seq_len 期望的KV序列长度，默认为0表示不检查
 * @return 如果需要创建新变量则返回true，否则返回false
 */
static inline bool needNewVar(VARP var, int axis, int seq_len, int kv_seq_len = 0) {
    // 如果变量为空，需要创建新变量
    if (var == nullptr) {
        return true;
    }

    // 如果指定维度的大小与期望序列长度不匹配，需要创建新变量
    if (var->getInfo()->dim[axis] != seq_len) {
        return true;
    }

    // 如果提供了KV序列长度且下一个维度大小不匹配，需要创建新变量
    if (kv_seq_len != 0 && var->getInfo()->dim[axis + 1] != kv_seq_len) {
        return true;
    }

    //所有条件都满足，不需要创建新变量
    return false;
}

/**
 * @brief 获取输入token序列的嵌入表示
 * @param input_ids 输入的token ID序列
 * @return 返回形状为[seq_len, 1, hidden_size]的嵌入向量
 */
VARP Llm::embedding(const std::vector<int>& input_ids) {
    AUTOTIME;
    int hidden_size = mConfig->hidden_size();
    int seq_len = static_cast<int>(input_ids.size());

    // 创建输出张量，形状为[序列长度, 1, 隐藏层大小]
    VARP res = _Input({seq_len, 1, hidden_size}, NCHW);
    
    // 使用磁盘嵌入层将token ID转换为嵌入向量，以节省内存
    mDiskEmbedding->embedding(input_ids, res->writeMap<float>());
    return res;
}

/**
 * @brief 将token ID解码为对应的字符串
 * @param id 需要解码的token ID
 * @return 解码后的字符串
 */
std::string Llm::tokenizer_decode(int id) {
    std::string word = mTokenizer->decode(id);
    
    // 修复UTF-8乱码字符：如果解码结果是形如<0xFF>的格式，则将其转换为对应的字符
    if (word.length() == 6 && word[0] == '<' && word[word.length() - 1] == '>' && word[1] == '0' && word[2] == 'x') {
        int num = std::stoi(word.substr(3, 2), nullptr, 16);
        word    = static_cast<char>(num);
    }
    return word;
}

/**
 * @brief 生成注意力掩码（attention mask），用于控制模型在自回归生成过程中对历史 token 的关注范围。
 *
 * 根据配置中的 attention_mask 类型和 attention_type，生成不同类型的注意力掩码：
 * - 支持 float 类型的全注意力和滑动窗口混合注意力；
 * - 支持 int 类型的 GLM、GLM2 等特殊掩码格式；
 * - 对于 float 类型，还支持缓存机制以节省内存。
 *
 * @param seq_len 当前输入序列的长度（通常是新生成的 token 数量）。
 * @return 返回生成的注意力掩码变量（VARP 类型）。
 */
VARP Llm::gen_attention_mask(int seq_len) {
    int kv_seq_len = mContext->all_seq_len + seq_len;

    //处理 float 类型的注意力掩码
    if (mConfig->attention_mask() == "float") {

        // 混合注意力类型：同时支持 full 和 sliding window 注意力
        if (mConfig->attention_type() == "mix") {
            const int sliding_window = mConfig->sliding_window();
            
            // 创建混合注意力掩码张量，形状为 [2, 1, 1, seq_len, kv_seq_len]
            // 第一个维度分别表示 full attention 和 sliding window attention，即0表示full，1表示sliding window
            attentionMask = _Input({2, 1, 1, seq_len, kv_seq_len}, NCHW, halide_type_of<float>());
            auto full_attn_ptr = attentionMask->writeMap<float>();
            
            // 构造 full attention 掩码：每个 query 只能看到其位置及之前的 key
            for (int i = 0; i < seq_len; i++) {
                const int query_pos = i + (kv_seq_len - seq_len);
                for (int j = 0; j < kv_seq_len; j++) {
                    if (j > query_pos) {
                        full_attn_ptr[kv_seq_len * i + j] = std::numeric_limits<float>::lowest();
                    } else {
                        full_attn_ptr[kv_seq_len * i + j] = 0.0f;
                    }
                }
            }
            
            // 构造 sliding window attention 掩码：限制 key 在滑动窗口范围内
            auto sliding_attn_ptr = full_attn_ptr + seq_len * kv_seq_len;
            const int query_pos_offset = kv_seq_len - seq_len;
            for (int i = 0; i < seq_len; i++) {
                const int query_pos = i + query_pos_offset;
                for (int j = 0; j < kv_seq_len; j++) {
                    const int key_pos = j;
                    bool is_allowed = (key_pos <= query_pos) && (key_pos > query_pos - sliding_window);
                    if (is_allowed) {
                        sliding_attn_ptr[kv_seq_len * i + j] = 0.0f;
                    } else {
                        sliding_attn_ptr[kv_seq_len * i + j] = std::numeric_limits<float>::lowest();
                    }
                }
            }
            return attentionMask;
        }
                
        // 仅使用普通 square mask，适用于新生成 token 的情况，节省内存
        kv_seq_len = seq_len;

        // 如果已有缓存的 attention mask，则直接复用
        if (mAttentionMaskVarVec.size() > 0) {
            if(seq_len == 1) {
                return mAttentionMaskVarVec[0];
            }
            if (mAttentionMaskVarVec.size() > 1 && seq_len == mDraftLength) {
                return mAttentionMaskVarVec[1];
            }
        }

        // 否则创建新的 attention mask 张量
        attentionMask = _Input({1, 1, seq_len, kv_seq_len}, NCHW, halide_type_of<float>());
        auto ptr = attentionMask->writeMap<float>();
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < kv_seq_len; j++) {
                ptr[kv_seq_len * i + j] = (j > i) * std::numeric_limits<float>::lowest();
            }
        }
        return attentionMask;
    } else { // 处理 int 类型的注意力掩码

        //判断是否需要创建新的变量
        if (needNewVar(attentionMask, 2, seq_len, kv_seq_len)) {
            attentionMask = _Input({1, 1, seq_len, kv_seq_len}, NCHW, halide_type_of<int>());
        } else {
            return attentionMask;
        }
        auto ptr = attentionMask->writeMap<int>();

        // 特殊处理 chatglm类型的掩码
        if (mConfig->attention_mask() == "glm") {
            // chatglm
            for (int i = 0; i < seq_len * kv_seq_len; i++) {
                ptr[i] = 0;
            }
            if (seq_len > 1) {
                for (int i = 1; i < seq_len; i++) {
                    ptr[seq_len * i - 1] = 1;
                }
            }
        } else { // 处理GLM2 或者其它int类型掩码
            bool is_glm2 = mConfig->attention_mask() == "glm2";
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < kv_seq_len; j++) {
                    int row              = i + mContext->all_seq_len;
                    ptr[seq_len * i + j] = is_glm2 ? j > row : j <= row;
                }
            }
        }
        return attentionMask;
    }
}

/**
 * @brief 生成位置ID（position ids）张量，用于模型推理时的位置编码。
 * 
 * 根据不同的 attention_mask 类型（如 "glm" 或 "glm2"）和序列长度，
 * 构造相应的位置ID张量。该函数支持多种模型结构（如 ChatGLM 系列），
 * 并根据上下文维护的序列长度信息生成合适的位置索引。
 * 
 * @param seq_len 当前输入序列的长度。
 * @return 返回一个包含位置ID的张量变量（VARP类型）。
 */
VARP Llm::gen_position_ids(int seq_len) {
    if (mConfig->attention_mask() == "glm") {
        // 处理 ChatGLM 模型的位置ID生成逻辑
        if (needNewVar(positionIds, 2, seq_len)) {
            positionIds = _Input({1, 2, seq_len}, NCHW, halide_type_of<int>());
        }
        auto ptr = positionIds->writeMap<int>();
        if (seq_len == 1) {
            // 处理单步推理,即单token推理阶段的位置ID设置
            ptr[0] = mContext->all_seq_len - mContext->gen_seq_len - 2;
            ptr[1] = mContext->gen_seq_len + 1;
        } else {
            // 处理多步推理,即多token推理阶段位置ID设置
            for (int i = 0; i < seq_len - 1; i++) {
                ptr[i]           = i;
                ptr[seq_len + i] = 0;
            }
            //设置最后一个位置ID
            ptr[seq_len - 1]     = seq_len - 2;
            ptr[2 * seq_len - 1] = 1;
        }
        return positionIds;
    } else { // 处理非 ChatGLM 模型（如 GLM-2）的位置ID生成逻辑
        bool is_glm2 = mConfig->attention_mask() == "glm2";
        if (seq_len == 1) { 
            // 处理单步推理，即单token推理情况下的位置ID设置
            auto ptr = mPositionIdsVarVec[0]->writeMap<int>();
            ptr[0] = is_glm2 ? mContext->gen_seq_len : mContext->all_seq_len;
            return mPositionIdsVarVec[0];
        }

        // 处理多步推理：如果存在预分配的 draft length 张量且当前长度匹配，则复用
        if(mPositionIdsVarVec.size() > 1 && seq_len == mDraftLength) {
            auto ptr = mPositionIdsVarVec[1]->writeMap<int>();
            for (int i = 0; i < seq_len; i++) {
                ptr[i] = i + mContext->all_seq_len;
            }
            return mPositionIdsVarVec[1];
        }
        
        // 动态创建新的位置ID张量
        positionIds = _Input({seq_len}, NCHW, halide_type_of<int>());
        auto ptr = positionIds->writeMap<int>();
        if (seq_len == 1) {
            // 再次处理单token情况（冗余但保证兼容性）
            ptr[0] = is_glm2 ? mContext->gen_seq_len : mContext->all_seq_len;
        } else {
            // 填充完整序列的位置ID
            for (int i = 0; i < seq_len; i++) {
                ptr[i] = i + mContext->all_seq_len;
            }
        }
        return positionIds;
    }
}

/**
 * @brief 判断给定的token是否为停止符
 * 
 * 该函数用于检查指定的token ID是否表示一个停止符，
 * 通过调用tokenizer对象的is_stop方法来实现判断逻辑。
 * 
 * @param token_id 要检查的token标识符
 * @return bool 如果是停止符则返回true，否则返回false
 */
bool Llm::is_stop(int token_id) {
    return mTokenizer->is_stop(token_id);
}
} // namespace Transformer
} // namespace MNN
