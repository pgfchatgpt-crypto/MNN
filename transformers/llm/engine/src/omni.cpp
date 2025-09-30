//
//  omni.cpp
//
//  Created by MNN on 2025/04/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif
#include <regex>
#include <algorithm>
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include "omni.hpp"
#include "kvmeta.hpp"
#include "llmconfig.hpp"
#include "tokenizer.hpp"
#include "diskembedding.hpp"
#include "sampler.hpp"
#include "httplib.h"
#ifdef LLM_SUPPORT_VISION
#include <cv/cv.hpp>
#endif
#ifdef LLM_SUPPORT_AUDIO
#include <audio/audio.hpp>
#endif

namespace MNN {
using namespace Express;
namespace Transformer {

template <typename T>
static inline VARP _var(std::vector<T> vec, const std::vector<int> &dims) {
    return _Const(vec.data(), dims, NHWC, halide_type_of<T>());
}

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

/**
 * @brief Omni类的构造函数，用于初始化Omni模型实例
 * @param config 指向LlmConfig配置对象的智能指针，包含模型配置信息
 * 
 * 该构造函数首先调用基类Llm的构造函数，然后根据配置信息初始化视觉和音频相关的参数。
 * 对于视觉模型，会从配置中读取图像处理相关参数；对于音频模型，目前为空实现。
 */
Omni::Omni(std::shared_ptr<LlmConfig> config) : Llm(config) {
    // 初始化视觉相关参数
    if (config->is_visual()) {
        mVisionHeight = config->config_.value("image_size", mVisionHeight);
        mVisionWidth  = mVisionHeight;
        mVisionPad    = config->config_.value("image_pad", mVisionPad);
        mVisionStart  = config->config_.value("vision_start", mVisionStart);
        mVisionEnd    = config->config_.value("vision_end", mVisionEnd);
        mVisionMean   = config->config_.value("image_mean", mVisionMean);
        mVisionNorm   = config->config_.value("image_norm", mVisionNorm);
        mVisionSizeUnit = config->config_.value("image_size_unit", mVisionSizeUnit);
        mVisionMaxSize = config->config_.value("image_max_size", mVisionMaxSize);
        mVisionGlobal = config->config_.value("global_image", mVisionGlobal);
    }

    // 初始化音频相关参数
    if (config->is_audio()) {}
}

/**
 * @brief 加载Omni模型及其相关组件。
 *
 * 该函数首先调用基类Llm的load方法进行基础加载操作。然后根据配置判断是否需要加载Talker模块，
 * 若需要则创建并加载Talker对象。接着根据配置决定运行时管理器（RuntimeManager）是复用现有实例
 * 还是创建一个新的实例，并设置相应的后端配置（如线程数、功耗、内存、精度等）。最后根据配置加载
 * 视觉模块和音频模块。
 */
void Omni::load() {
    Llm::load();
    if (mConfig->has_talker()) {
        mTalker.reset(new Talker(mConfig, this));
        mTalker->load();
    }
    ScheduleConfig config;
    if (mConfig->mllm_config_.empty()) {
        mProcessorRuntimeManager = mRuntimeManager;
    } else {
        BackendConfig cpuBackendConfig;
        config.type      = backend_type_convert(mConfig->backend_type(true));
        config.numThread = mConfig->thread_num(true);
        if(config.type == 3){
            config.numThread |= 64;
        }
        if (mConfig->power(true) == "high") {
            cpuBackendConfig.power = BackendConfig::Power_High;
        } else if (mConfig->power(true) == "low") {
            cpuBackendConfig.power = BackendConfig::Power_Low;
        }
        if (mConfig->memory(true) == "high") {
            cpuBackendConfig.memory = BackendConfig::Memory_High;
        } else if (mConfig->memory(true) == "low") {
            cpuBackendConfig.memory = BackendConfig::Memory_Low;
        }
        if (mConfig->precision(true) == "high") {
            cpuBackendConfig.precision = BackendConfig::Precision_High;
        } else if (mConfig->precision(true) == "low") {
            cpuBackendConfig.precision = BackendConfig::Precision_Low;
        }
        config.backendConfig = &cpuBackendConfig;
        mProcessorRuntimeManager.reset(Executor::RuntimeManager::createRuntimeManager(config));
        setRuntimeHint(mProcessorRuntimeManager);
    }
    Module::Config module_config;
    if(config.type == MNN_FORWARD_NN) {
        module_config.shapeMutable = false;
        module_config.rearrange    = false;
    } else {
        module_config.shapeMutable = true;
        module_config.rearrange    = true;
    }
    if (mConfig->is_visual()) {
        mVisionModule.reset(Module::load({}, {}, mConfig->visual_model().c_str(), mProcessorRuntimeManager, &module_config));
    }
    if (mConfig->is_audio()) {
        mAudioModule.reset(Module::load({}, {}, mConfig->audio_model().c_str(), mProcessorRuntimeManager, &module_config));
    }
}

#ifdef LLM_SUPPORT_VISION
/**
 * @brief 默认视觉处理函数，对输入图像进行预处理、特征提取并生成对应的ID序列
 * @param image 输入的图像变量
 * @return std::vector<int> 图像对应的ID序列，包含填充ID，以及可能的开始和结束标记
 */
std::vector<int> Omni::defaultVisionProcess(VARP image) {
    // 调整图像高度和宽度为指定单元的倍数
    mVisionHeight = UP_DIV(mVisionHeight, mVisionSizeUnit) * mVisionSizeUnit;
    mVisionWidth  = UP_DIV(mVisionWidth, mVisionSizeUnit) * mVisionSizeUnit;

    // 图像预处理：调整大小、颜色空间转换、归一化等
    image = MNN::CV::resize(image, {mVisionWidth, mVisionHeight}, 0, 0,
                            MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                            mVisionMean, mVisionNorm);

    // 增加批次维度并转换数据格式
    image = Express::_Unsqueeze(image, {0});
    image = Express::_Convert(image, NC4HW4);

    // 通过视觉模块前向传播获取图像嵌入特征
    auto imageEmbedding = mVisionModule->forward(image);

    // 保存图像嵌入特征并计算其长度
    mVisionEmbeddings.push_back(imageEmbedding);
    int visionLen = imageEmbedding->getInfo()->dim[0];

    // 创建图像ID序列，初始用填充ID填充
    std::vector<int> imgIds(visionLen, mVisionPad);

    // 如果设置了开始和结束标记，则在序列首尾添加相应标记
    if (mVisionStart >= 0 && mVisionEnd >= 0) {
        imgIds.insert(imgIds.begin(), mVisionStart);
        imgIds.push_back(mVisionEnd);
    }
    return imgIds;
}

/**
 * @brief Qwen2-VL / Qwen2.5-VL 模型处理图像输入，生成视觉模型所需的嵌入向量及对应的标识符序列。
 * 
 * 该函数对输入图像进行预处理，包括尺寸调整、归一化、分块等操作，
 * 然后通过视觉模块提取图像嵌入，并构建位置信息和注意力掩码。
 * 最终返回图像嵌入对应的标识符序列（包含开始、结束和填充标记）。
 * 
 * @param image 输入图像变量（MNN::Express::VARP 类型）
 * @return std::vector<int> 图像嵌入对应的标识符序列，包含开始、结束和填充标记
 */
std::vector<int> Omni::qwen2VisionProcess(VARP image) {
    const auto inputNames = mVisionModule->getInfo()->inputNames;
    bool hasWindowIndex = inputNames.size() == 4 && inputNames[3] == "window_index";
    
    // 调整图像尺寸为28的倍数，适配视觉模型输入要求
    mVisionHeight = round(mVisionHeight / 28.0) * 28;
    mVisionWidth = round(mVisionWidth / 28.0) * 28;

    // 图像预处理：调整大小、颜色空间转换、归一化
    image = MNN::CV::resize(image, {mVisionWidth, mVisionHeight}, 0, 0,
                            MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                            mVisionMean, mVisionNorm);
    
    // 增加批次维度并转换数据格式NCHW                            
    image = Express::_Unsqueeze(image, {0});
    image = Express::_Convert(image, NCHW);

    // 构造图像块（patches），复制一份用于后续处理
    auto patches = Express::_Concat({image, image}, 0);
    auto patches_dim = patches->getInfo()->dim;
    int temporal = patches_dim[0];
    int channel  = patches_dim[1];
    int height   = patches_dim[2];
    int width    = patches_dim[3];

    // 定义图像分块相关参数
    constexpr int temporal_patch_size = 2;
    constexpr int patch_size = 14;
    constexpr int merge_size = 2;

    // 计算时间、高度、宽度方向上的网格数
    int grid_t = temporal / temporal_patch_size;
    int grid_h = height / patch_size;
    int grid_w = width / patch_size;
    
    // 添加位置ID信息
    addPositionIds(grid_t, grid_h / merge_size, grid_w / merge_size);

    // 对图像块进行重排和重塑，构建视觉token序列
    patches = Express::_Reshape(patches, {
        grid_t, temporal_patch_size,
        channel,
        grid_h / merge_size, merge_size, patch_size,
        grid_w / merge_size, merge_size, patch_size,
    });
    // 重排和重塑,矩阵转置矩阵
    patches = Express::_Permute(patches, {0, 3, 6, 4, 7, 2, 1, 5, 8}); // N, T, C, H, W, H, W, H, W
    patches = Express::_Reshape(patches, {
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size
    });
    const int seq_len = grid_t * grid_h * grid_w;

    // 构建位置ID张量
    const int wblock_size = merge_size * merge_size;
    const int hblock_size = wblock_size * grid_w / merge_size;
    VARP position_ids = Express::_Input({2, seq_len}, NCHW, halide_type_of<int>());
    auto hpos_ptr = position_ids->writeMap<int>();
    auto wpos_ptr = hpos_ptr + seq_len;

    // 填充位置ID数据
    for (int i = 0; i < grid_h; i++) {
        int h_idx = i / merge_size, h_off = i % merge_size;
        for (int j = 0; j < grid_w; j++) {
            int w_idx = j / merge_size, w_off = j % merge_size;
            int index = h_idx * hblock_size + w_idx * wblock_size + h_off * 2 + w_off;
            hpos_ptr[index] = i;
            wpos_ptr[index] = j;
        }
    }
    VARP attention_mask, window_index;
    VARPS moduleInputs= {patches, position_ids};

    // 如果模型支持窗口索引，则构造窗口索引和注意力掩码
    if (hasWindowIndex) {
        // 构建窗口索引
        window_index = Express::_Input({seq_len / 4}, NCHW, halide_type_of<int>());
        auto window_index_ptr = window_index->writeMap<int>();
        const int merge_unit = merge_size * merge_size;
        const int vit_merger_window_size = 4;
        int llm_grid_h = grid_h / merge_size;
        int llm_grid_w = grid_w / merge_size;
        int pad_h = vit_merger_window_size - (llm_grid_h % vit_merger_window_size);
        int pad_w = vit_merger_window_size - (llm_grid_w % vit_merger_window_size);
        int new_h = llm_grid_h + pad_h; // 划窗的高度
        int new_w = llm_grid_w + pad_w; // 划窗的宽度
        int num_windows_h = new_h / vit_merger_window_size;
        int num_windows_w = new_w / vit_merger_window_size;
        std::vector<int> seqlens;
        int window_index_idx = 0;

        // 填充窗口索引数据 [T,H,W]
        for (int t = 0; t < grid_t; ++t) {
            for (int win_h = 0; win_h < num_windows_h; ++win_h) {
                for (int win_w = 0; win_w < num_windows_w; ++win_w) {
                    int count = 0;
                    for (int i = 0; i < vit_merger_window_size; ++i) {
                        int h_global = win_h * vit_merger_window_size + i;
                        if (h_global >= llm_grid_h) continue;
                        for (int j = 0; j < vit_merger_window_size; ++j) {
                            int w_global = win_w * vit_merger_window_size + j;
                            if (w_global >= llm_grid_w) continue;
                            int idx = t * llm_grid_h * llm_grid_w + h_global * llm_grid_w + w_global;
                            window_index_ptr[window_index_idx++] = idx;
                            ++count;
                        }
                    }
                    seqlens.push_back(count);
                }
            }
        }

        // 构建窗口序列长度累计数组，flatten
        std::vector<int> cu_window_seqlens = {0};
        int prev = cu_window_seqlens.back();
        for (int s : seqlens) {
            cu_window_seqlens.push_back(prev + s * merge_unit);
            prev = cu_window_seqlens.back();
        }

        // 构建注意力掩码
        attention_mask = Express::_Input({2, 1, seq_len, seq_len}, NCHW);
        auto attention_mask_ptr = attention_mask->writeMap<float>();
        ::memset(attention_mask_ptr, 0, seq_len * seq_len * sizeof(float));
        attention_mask_ptr = attention_mask_ptr + seq_len * seq_len;
        for (int i = 0; i < seq_len * seq_len; i++) {
            attention_mask_ptr[i] = std::numeric_limits<float>::lowest();
        }

        // 设置窗口内注意力可见区域
        for (size_t i = 1; i < cu_window_seqlens.size(); ++i) {
            for (int j = cu_window_seqlens[i - 1]; j < cu_window_seqlens[i]; ++j) {
                for (int k = cu_window_seqlens[i - 1]; k < cu_window_seqlens[i]; ++k) {
                    attention_mask_ptr[seq_len * j + k] = 0;
                }
            }
        }
        moduleInputs.push_back(attention_mask);
        moduleInputs.push_back(window_index);
    } else {
        // 构建普通注意力掩码
        attention_mask = Express::_Input({1, seq_len, seq_len}, NCHW);
        ::memset(attention_mask->writeMap<float>(), 0, seq_len * seq_len * sizeof(float));
        moduleInputs.push_back(attention_mask);
    }
#ifdef DEBUG_IMAGE
    // 调试模式下保存输入张量
    patches.fix(MNN::Express::VARP::CONSTANT);
    patches->setName("patches");
    position_ids.fix(MNN::Express::VARP::CONSTANT);
    position_ids->setName("position_ids");
    attention_mask.fix(MNN::Express::VARP::CONSTANT);
    attention_mask->setName("attention_mask");
    MNN::Express::Variable::save({patches, position_ids, attention_mask}, "input.mnn");
#endif

    // 执行视觉模块前向推理，获取图像嵌入
    auto imageEmbedding = mVisionModule->onForward(moduleInputs)[0];
#ifdef DEBUG_IMAGE
    //调试模式下保存输出张量
    imageEmbedding->setName("image_embeds");
    MNN::Express::Variable::save({imageEmbedding}, "output.mnn");
#endif
    // 将图像嵌入保存到全局列表中
    mVisionEmbeddings.push_back(imageEmbedding);

    // 构建图像嵌入对应的标识符序列
    int visionLen = imageEmbedding->getInfo()->dim[0];
    std::vector<int> imgIds(visionLen, mVisionPad);
    imgIds.insert(imgIds.begin(), mVisionStart);
    imgIds.push_back(mVisionEnd);
    return imgIds;
}

/**
 * @brief 对输入图像进行视觉处理，生成对应的视觉嵌入向量和图像标识符序列。
 *
 * 该函数使用SmolVLM模型对图像进行预处理与特征提取。若图像尺寸超过设定单元大小，
 * 则将其划分为多个局部图像块，并分别处理；同时保留一个全局缩放版本用于整体理解。
 * 最终输出包括所有图像块及全局图像的嵌入表示以及对应的语言模型token ID序列。
 *
 * @param image 输入图像变量（VARP类型），通常为BGR格式的图像数据。
 * @return std::vector<int> 图像相关的token ID序列，供语言模型解析使用。
 */
std::vector<int> Omni::smolvlmVisionProcess(VARP image) {
    //定义每个图像块中视觉token的数量
    constexpr int visionLen = 64;

    // 判断是否需要将图像分割成多个patch
    bool splitImage = mVisionHeight > mVisionSizeUnit || mVisionWidth > mVisionSizeUnit;

    // 预处理全局图像：调整尺寸、颜色空间转换、归一化并增加batch维度
    auto globalImage = MNN::CV::resize(image, {mVisionSizeUnit, mVisionSizeUnit}, 0, 0,
                                       MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                                       mVisionMean, mVisionNorm);
    globalImage = Express::_Unsqueeze(globalImage, {0});
    globalImage = Express::_Convert(globalImage, NCHW);
    std::vector<int> imgIds;
    
    // 如果图像过大，需要切分处理
    if (splitImage) {

        // 调整图像高度和宽度到最近的mVisionSizeUnit倍数
        mVisionHeight = round(mVisionHeight / (float)mVisionSizeUnit) * mVisionSizeUnit;
        mVisionWidth = round(mVisionWidth / (float)mVisionSizeUnit) * mVisionSizeUnit;

        // 控制最大尺寸不超过限制
        if (mVisionHeight > mVisionMaxSize) {
            mVisionHeight = mVisionMaxSize;
        }
        if (mVisionWidth > mVisionMaxSize) {
            mVisionWidth = mVisionMaxSize;
        }

        // 缩放原图以适配新的尺寸
        auto patches = MNN::CV::resize(image, {mVisionWidth, mVisionHeight}, 0, 0,
                                       MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                                       mVisionMean, mVisionNorm);
        patches = Express::_Unsqueeze(patches, {0});
        patches = Express::_Convert(patches, NCHW);
        
        // 获取图像维度信息
        auto imageDims = patches->getInfo()->dim;
        int batch    = imageDims[0];
        int channel  = imageDims[1];
        int height   = imageDims[2];
        int width    = imageDims[3];

        // 计算网格划分数量
        int grid_h = height / mVisionSizeUnit;
        int grid_w = width / mVisionSizeUnit;

        // 将图像reshape为[batch, channel, grid_h, mVisionSizeUnit, grid_w, mVisionSizeUnit]六维张量以便于重排
        patches = Express::_Reshape(patches, {
            batch,
            channel,
            grid_h, mVisionSizeUnit,
            grid_w, mVisionSizeUnit,
        });

        // 变换轴顺序，使得每个patch连续排序
        patches = Express::_Permute(patches, {0, 2, 4, 1, 3, 5});

        // reshape为（batch * grid_h * grid_w, C, H, W）
        patches = Express::_Reshape(patches, {
            batch * grid_h * grid_w,
            channel,
            mVisionSizeUnit,
            mVisionSizeUnit
        });

        // 拼接局部patches和全局图像作为最终输入
        patches = _Concat({patches, globalImage}, 0);

        // 前向传播获取图像嵌入
        auto imageEmbedding = mVisionModule->forward(patches);

        // 提取每个patch的embedding，然后使用动态的注意力机制存入成员变量
        auto embeddingDims = imageEmbedding->getInfo()->dim;
        for (int i = 0; i < embeddingDims[0]; i++) {
            auto embedding = _Squeeze(_GatherV2(imageEmbedding, _var<int>({i}, {1}), _var<int>({0}, {1})), {0});
            mVisionEmbeddings.push_back(embedding);
        }

        // 构造每一块图像的位置标记及其对应的token IDs
        int endRow = tokenizer_encode("\n")[0];
        for (int h = 0; h < grid_h; h++) {
            for (int w = 0; w < grid_w; w++) {
                imgIds.push_back(mVisionStart);
                // <row_{h+1}_col{w+1}>
                std::string image_pos = "<row_" + std::to_string(h + 1) + "_col_" + std::to_string(w + 1) + ">";
                imgIds.push_back(tokenizer_encode(image_pos)[0]);
                for (int p = 0; p < visionLen; p++) {
                    imgIds.push_back(mVisionPad);
                }
            }
            imgIds.push_back(endRow);
        }
        imgIds.push_back(endRow);
    } else { //不需要切分时仅处理全局图像
        auto imageEmbedding = mVisionModule->forward(globalImage);
        mVisionEmbeddings.push_back(_Squeeze(imageEmbedding, {0}));
    }

    // 添加全局图像的token ID标识
    imgIds.push_back(mVisionStart);
    imgIds.push_back(mVisionGlobal);
    for (int p = 0; p < visionLen; p++) {
        imgIds.push_back(mVisionPad);
    }
    imgIds.push_back(mVisionEnd);
    return imgIds;
}

/**
 * @brief 根据原始图像尺寸和patch大小，计算最佳的图像缩放尺寸、细化尺寸以及最优网格划分。
 *
 * 该函数用于视觉模型中对输入图像进行预处理时的尺寸选择。它会根据图像面积与目标分辨率的比例，
 * 确定合适的切片数量，并在候选网格中选择最接近原始宽高比的一个。然后基于该网格计算出最佳的
 * patch尺寸和整体图像尺寸。
 *
 * @param original_size 原始图像的尺寸，first为高度，second为宽度。
 * @param patch_size 每个patch的尺寸（假设为正方形）。
 * @return 返回一个包含三个元素的向量：
 *         - 第一个元素是源图像的目标尺寸（source_image_size）；
 *         - 第二个元素是细化后的图像尺寸（refine_image_size）；
 *         - 第三个元素是最优的网格划分（grid_h, grid_w）。
 */
std::vector<std::pair<int, int>> minicpmBestSize(std::pair<int, int> original_size, int patch_size) {
    constexpr int max_slice_nums = 9, scale_resolution = 448;
    
    // 定义一个lambda函数，用于根据给定尺寸和是否放大来计算目标尺寸
    auto _get_target_size =
        [&](std::pair<int, int> size, bool upscale) -> std::pair<int, int> {
        int h = size.first;
        int w = size.second;
        int target_w, target_h;

        // 如果不需要放大且图像面积小于等于目标面积，则保持原尺寸
        if (!upscale && (static_cast<long long>(w) * h <= static_cast<long long>(scale_resolution) * scale_resolution)) {
            target_w = w;
            target_h = h;
        } else {
            // 否则按比例缩放到目标分辨率范围内
            double r = (h != 0) ? static_cast<double>(w) / h : 0.0;
            if (r > 0) {
                target_h = static_cast<int>(scale_resolution / std::sqrt(r));
                target_w = static_cast<int>(target_h * r);
            } else {
                target_h = 0;
                target_w = scale_resolution;
            }
        }

        // 将目标尺寸调整为patch_size的整数倍，最小为patch_size
        int final_h = std::max(static_cast<int>(std::round(static_cast<double>(target_h) / patch_size)) * patch_size, patch_size);
        int final_w = std::max(static_cast<int>(std::round(static_cast<double>(target_w) / patch_size)) * patch_size, patch_size);
        return std::make_pair(final_h, final_w);
    };
    int original_height = original_size.first;
    int original_width = original_size.second;

    // 计算图像面积与目标面积的比例，决定切片数量
    double ratio = (static_cast<double>(original_width) * original_height) / (static_cast<double>(scale_resolution) * scale_resolution);
    int multiple = std::min(static_cast<int>(std::ceil(ratio)), max_slice_nums);

    // 构造候选切片数量集合：当前值及其前后各一个
    std::vector<std::pair<int, int>> candidates;
    std::set<int> nums_to_check;
    if (multiple > 1) nums_to_check.insert(multiple - 1);
    nums_to_check.insert(multiple);
    nums_to_check.insert(multiple + 1);

    // 遍历所有候选切片数量，生成所有可能的网格划分(m x n = num)
    for (std::set<int>::iterator it = nums_to_check.begin(); it != nums_to_check.end(); ++it) {
        int num = *it;
        if (num >= 1 && num <= max_slice_nums) {
            for (int m = 1; m * m <= num; ++m) {
                if (num % m == 0) {
                    candidates.push_back(std::make_pair(m, num / m));
                    if (m * m != num) candidates.push_back(std::make_pair(num / m, m));
                }
            }
        }
    }

    // 如果没有找到候选网格，则默认使用1x1网格
    if (candidates.empty()) { candidates.push_back(std::make_pair(1, 1)); }
    
    // 计算原始图像的对数宽高比
    double log_ratio = std::log(static_cast<double>(original_width) / original_height);
    
    // 在候选网格中选择最接近原始宽高比的网格
    std::pair<int, int> best_grid = *std::min_element(candidates.begin(), candidates.end(),
        [log_ratio](const std::pair<int, int>& g1, const std::pair<int, int>& g2) {
            auto key = [log_ratio](const std::pair<int, int>& g) -> double {
                if (g.first == 0) return std::numeric_limits<double>::infinity();
                return std::abs(log_ratio - std::log(static_cast<double>(g.second) / g.first));
            };
            return key(g1) < key(g2);
        });

    // 获取源图像的目标尺寸（不放大）
    std::pair<int, int> source_image_size = _get_target_size(original_size, false);

    // 根据最优网格划分计算每个patch的理想尺寸
    double patch_h = static_cast<double>(original_height) / best_grid.first;
    double patch_w = static_cast<double>(original_width) / best_grid.second;

    // 获取每个patch的目标尺寸（允许放大）
    std::pair<int, int> best_patch_size = _get_target_size(std::make_pair(static_cast<int>(patch_h), static_cast<int>(patch_w)), true);
    
    // 计算细化后的整体图像尺寸
    std::pair<int, int> refine_image_size = std::make_pair(
        best_patch_size.first * best_grid.first,
        best_patch_size.second * best_grid.second
    );

    // 构造并返回结果
    std::vector<std::pair<int, int>> result;
    result.push_back(source_image_size);
    result.push_back(refine_image_size);
    result.push_back(best_grid);
    return result;
}


/**
 * @brief minicpm处理图像输入并生成视觉嵌入及对应的token IDs。
 *
 * 此函数接收一个图像变量（VARP），对其进行预处理、切片、编码等操作，
 * 最终输出用于模型输入的图像token ID序列。该过程包括图像缩放、分块、位置ID计算、注意力掩码构建等步骤。
 *
 * @param image 输入图像变量，类型为Express::VARP。
 * @return 返回表示图像信息的整数ID列表，可用于后续文本-图像联合建模。
 */
std::vector<int> Omni::minicpmVisionProcess(VARP image) {
    constexpr int visionLen = 64, patchesPerSide = 70;
    const int patchSize = mVisionSizeUnit;

    // 计算最优尺寸配置：全局大小、细化大小以及切片网格数量
    auto bestSize = minicpmBestSize(std::make_pair(mVisionHeight, mVisionWidth), patchSize);
    auto globalSize = bestSize[0];
    auto refineSize = bestSize[1];
    auto sliceGrids = bestSize[2];

    // 定义图像重排Lambda函数：调整图像到目标尺寸，并按指定网格划分成子图块
    auto reoderImage = [this, &patchSize](
        Express::VARP img, std::pair<int, int> targetSize, std::pair<int,int> grid, std::vector<int>& tgtSize) {
        // 调整图像大小并转换颜色空间与归一化方式
        auto patches = MNN::CV::resize(img, {targetSize.second, targetSize.first}, 0, 0,
                                    MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                                    mVisionMean, mVisionNorm);
        patches = Express::_Unsqueeze(patches, {0});
        patches = Express::_Convert(patches, NCHW);
        
        // 获取图像维度信息
        auto imageDims = patches->getInfo()->dim;
        int batch   = imageDims[0];
        int channel = imageDims[1];
        int height  = imageDims[2];
        int width   = imageDims[3];
        int gridH   = grid.first;
        int gridW   = grid.second;
        int subHeight = height / gridH;
        int subWidth = width / gridW;
        int numPatchesH = subHeight / patchSize;
        int numPatchesW = subWidth / patchSize;

        // 对patches进行reshape 和 permute以重新组织数据结构
        patches = Express::_Reshape(patches, {
            channel,
            gridH,
            numPatchesH,
            patchSize,
            gridW,
            numPatchesW,
            patchSize
        });
        patches = Express::_Permute(patches, {1, 4, 0, 3, 2, 5, 6}); // NCHW -> NLC 维度重组
        
        // patches重新组织为NLC， 以便于后续拼接使用
        patches = Express::_Reshape(patches, {
            gridH * gridW,
            channel,
            patchSize,
            numPatchesH * numPatchesW * patchSize
        });

        // 将每个patch的高度记录进tgtSize中
        for (int i = 0; i < gridH * gridW; i++) {
            tgtSize.push_back(numPatchesH);
            tgtSize.push_back(numPatchesW);
        }
        return patches;
    };
    
    // 存储每张图片的patch数量
    std::vector<int> tgtSize;
    
    // 分别获取全局图像和局部细化图像的patches
    auto globalImage = reoderImage(image, globalSize, std::make_pair(1, 1), tgtSize);
    auto refineImage = reoderImage(image, refineSize, sliceGrids, tgtSize);
    
    // 补齐globalImage在最后一个维度上使其与refineImage一致
    int globleDim = globalImage->getInfo()->dim[3];
    int refineDim = refineImage->getInfo()->dim[3];
    globalImage = _Pad(globalImage, _var<int>({0, 0, 0, 0, 0, 0, 0, refineDim - globleDim}, {8}), CONSTANT);
    
    // 拼接全局图像和细化图像作为最终像素值输入
    auto pixel_values = _Concat({globalImage, refineImage}, 0);

    // 构造position_ids: 根据patch的位置映射到固定范围内的索引
    int B = tgtSize.size() / 2;
    int S = tgtSize[0] * tgtSize[1]; //全局图像的patch数量
    int L = tgtSize[2] * tgtSize[3]; //细化图像单个patch区域中的patch数量
    auto position_ids = Express::_Input({B, L}, NCHW, halide_type_of<int>());
    auto posPtr = position_ids->writeMap<int>();
    memset(posPtr, 0, B * L * sizeof(int));
    for (int i = 0; i < B; ++i) {
        int nb_patches_h = tgtSize[i * 2];
        int nb_patches_w = tgtSize[i * 2 + 1];
        for (int h_idx = 0; h_idx < nb_patches_h; ++h_idx) {
            long bucket_h = static_cast<long>(std::floor(
                (static_cast<float>(h_idx) / nb_patches_h) * patchesPerSide
            ));
            for (int w_idx = 0; w_idx < nb_patches_w; ++w_idx) {
                long bucket_w = static_cast<long>(std::floor(
                    (static_cast<float>(w_idx) / nb_patches_w) * patchesPerSide
                ));
                long pos_id = bucket_h * patchesPerSide + bucket_w;
                long patch_idx = h_idx * nb_patches_w + w_idx;
                posPtr[i * L + patch_idx] = static_cast<int>(pos_id);
            }
        }
    }
    
    // 构建attention_mask：将非全局图像部分mask掉
    auto attention_mask = Express::_Input({B, L}, NCHW);
    auto maskPtr = attention_mask->writeMap<float>();
    memset(maskPtr, 0, B * L * sizeof(float));
    for (int i = S; i < L; i++) {
        maskPtr[i] = std::numeric_limits<float>::lowest();
    }

    // 构造tgt_sizes：存储各图像块的高宽信息
    auto tgt_sizes = Express::_Input({B, 2}, NCHW, halide_type_of<int>());
    ::memcpy(tgt_sizes->writeMap<int>(), tgtSize.data(), tgtSize.size() * sizeof(int));
    
    // 使用视觉模块前向传播得到图像embedding
    auto imageEmbedding = mVisionModule->onForward({pixel_values, position_ids, attention_mask, tgt_sizes})[0];
    
    // 提取各个图像块的embedding并保存至mVisionEmbeddings
    for (int i = 0; i < B; i++) {
        auto embedding = _Permute(_GatherV2(imageEmbedding, _var<int>({i}, {1}), _var<int>({0}, {1})), {1, 0, 2}); //动态注意力机制
        mVisionEmbeddings.push_back(embedding);
    }

    // 加载相关配置参数
    int visionSliceStart = mConfig->config_.value("vision_slice_start_id", 111);
    int visionSliceEnd = mConfig->config_.value("vision_slice_end_id", 112);
    int visionIdStart = mConfig->config_.value("vision_id_start_id", 113);
    int visionIdEnd = mConfig->config_.value("vision_id_end_id", 114);
    std::vector<int> imgIds;
    
    // 添加图像标识符起始标记
    imgIds.push_back(visionIdStart);

    // 编码当前图像编号并加入imgIds
    auto visionIdxIds = tokenizer_encode(std::to_string(mVisionNum));
    for (auto idx : visionIdxIds) {
        imgIds.push_back(idx);
    }
    imgIds.push_back(visionIdEnd);
    
    
    // 添加全局图像开始标记、填充tokens和结束标记
    imgIds.push_back(mVisionStart);
    for (int p = 0; p < visionLen; p++) {
        imgIds.push_back(mVisionPad);
    }
    imgIds.push_back(mVisionEnd);
    
    
    // 添加所有细化图像片段的标记
    for (int i = 0; i < B - 1; i++) {
        imgIds.push_back(visionSliceStart);
        for (int p = 0; p < visionLen; p++) {
            imgIds.push_back(mVisionPad);
        }
        imgIds.push_back(visionSliceEnd);
    }
    return imgIds;
}
#endif

/**
 * @brief 处理指定图像文件数据
 * @param file 图像文件路径
 * @return 处理结果，包含检测到的对象类别ID列表；如果不支持视觉处理则返回空向量
 */
std::vector<int> Omni::visionProcess(const std::string& file) {
#ifdef LLM_SUPPORT_VISION
    VARP image = MNN::CV::imread(file);
    return visionProcess(image);
#else
    return std::vector<int>(0);
#endif
}

/**
 * @brief 处理输入图像并返回图像标识符向量
 * @param image 输入的图像变量
 * @return std::vector<int> 图像处理后得到的标识符列表，如果处理失败或不支持视觉处理则返回空向量
 * 
 * 该函数根据不同的视觉模型输入要求，选择相应的图像处理方法。
 * 支持多种视觉模型的处理流程，包括qwen2、smolvlm、minicpm等。
 */
std::vector<int> Omni::visionProcess(VARP image) {
#ifdef LLM_SUPPORT_VISION
    if (image == nullptr) {
        MNN_PRINT("Omni Can't open image\n");
        return std::vector<int>(0);
    }
    Timer _t;
    std::vector<int> imgIds;
    const auto inputNames = mVisionModule->getInfo()->inputNames;
    if (inputNames.size() >= 3 && inputNames[0] == "patches") {
        imgIds = qwen2VisionProcess(image);
    } else if (inputNames[0] == "pixel_values") {
        if (inputNames.size() == 1) {
            imgIds = smolvlmVisionProcess(image);
        } else {
            imgIds = minicpmVisionProcess(image);
        }
    } else {
        imgIds = defaultVisionProcess(image);
    }
    mContext->vision_us += _t.durationInUs();
    // set vision number for image idx
    mVisionNum += 1;
    return imgIds;
#else
    return std::vector<int>(0);
#endif
}

/**
 * @brief 处理音频文件并返回处理结果
 * @param file 音频文件路径
 * @return std::vector<int> 音频处理结果，如果处理失败则返回空向量
 */
std::vector<int> Omni::audioProcess(const std::string& file) {
#ifdef LLM_SUPPORT_AUDIO
    constexpr int sample_rate = 16000;
    auto load_res        = MNN::AUDIO::load(file, sample_rate);
    VARP waveform        = load_res.first;
    if (waveform == nullptr) {
        MNN_PRINT("Omni Can't open audio: %s\n", file.c_str());
        return std::vector<int>(0);
    }
    return audioProcess(waveform);
#else
    return std::vector<int>(0);
#endif
}

/**
 * @brief 处理音频波形数据，生成对应的音频嵌入表示，并返回占位符ID列表。
 * 
 * 该函数接收一个音频波形变量作为输入，经过特征提取、模型推理等步骤，
 * 得到音频的嵌入向量，并将其保存至上下文。同时会根据音频长度生成对应数量的占位符ID。
 * 
 * @param waveform 输入的音频波形数据（MNN表达式变量）
 * @return std::vector<int> 与音频嵌入长度相同的占位符ID列表，用于后续处理
 */
std::vector<int> Omni::audioProcess(MNN::Express::VARP waveform) {
#ifdef LLM_SUPPORT_AUDIO
    if (waveform == nullptr) {
        MNN_PRINT("Omni Can't process audio: waveform is null\n");
        return std::vector<int>(0);
    }
    
    Timer _t;

    // 提取音频特征（如FBank特征）
    auto input_features  = MNN::AUDIO::whisper_fbank(waveform); //NCHW
    VARP audio_embedding;

    // 根据模型输入名称数量判断是否使用带注意力掩码的处理方式
    if (mAudioModule->getInfo()->inputNames.size() > 1) {
        // 计算序列长度并构造分段序列索引
        int seqlen = UP_DIV(input_features->getInfo()->dim[2], 2);
        constexpr int n_window = 100;
        std::vector<int> cu_seqlens;
        int curseq = 0;
        while (curseq < seqlen) {
            cu_seqlens.push_back(curseq);
            curseq += n_window;
        }
        if (seqlen % n_window != 0) {
            cu_seqlens.push_back(seqlen);
        }

        // 构造注意力掩码张量
        VARP attention_mask = _Input({1, seqlen, seqlen}, NCHW, halide_type_of<float>());
        auto ptr = attention_mask->writeMap<float>();
        for (int i = 0; i < seqlen; i++) {
            for (int j = 0; j < seqlen; j++) {
                ptr[seqlen * i + j] = std::numeric_limits<float>::lowest();
            }
        }

        // 设置局部窗口内的注意力权重为0
        for (size_t i = 1; i < cu_seqlens.size(); ++i) {
            for (int j = cu_seqlens[i - 1]; j < cu_seqlens[i]; ++j) {
                for (int k = cu_seqlens[i - 1]; k < cu_seqlens[i]; ++k) {
                    ptr[seqlen * j + k] = 0;
                }
            }
        }

        // 使用带注意力掩码的方式前向传播
        audio_embedding = mAudioModule->onForward({input_features, attention_mask})[0];
    } else {
        // Qwen2-Audio just support audio time <= 30s
        // 对于不支持长音频的模型(如Qwen2-Audio),限制音频时间不超过30s
        if (input_features->getInfo()->dim[2] > 3000) {
            input_features = _Slice(input_features, _var<int>({0, 0, 0}, {3}), _var<int>({-1, -1, 3000}, {3}));
        }

        // 前向传播获取音频嵌入向量
        audio_embedding = mAudioModule->forward(input_features);
    }

    // 调整维度顺序以适配模型输出格式
    audio_embedding = _Permute(audio_embedding, {1, 0, 2});

    // 记录音频处理耗时
    mContext->audio_us = _t.durationInUs();

    // 保存音频嵌入结果
    mAudioEmbeddings.push_back(audio_embedding);

    // 获取嵌入长度并更新位置ID
    int embed_len = audio_embedding->getInfo()->dim[0];
    addPositionIds(embed_len);

    // 返回与音频嵌入长度一致的占位符ID列表
    std::vector<int> audio_ids(embed_len, mAudioPad);
    return audio_ids;
#else
    // 如果未启用音频支持，则直接返回空向量
    return std::vector<int>(0);
#endif
}

/**
 * @brief 根据指定模式对输入信息进行多模态处理。
 *
 * 此函数根据传入的模式（如图像或音频）以及相关信息执行相应的处理流程。对于图像模式，
 * 它会解析图像尺寸信息，并可能从网络下载文件；之后依据配置决定是否调用视觉或音频处理函数。
 *
 * @param mode 处理模式，例如 "img" 表示图像处理，"audio" 表示音频处理。
 * @param info 包含待处理信息的字符串，可能是本地路径、URL 或带有元数据的信息。
 * @return std::vector<int> 视觉或音频处理的结果向量，若未触发有效处理则返回空向量。
 */
std::vector<int> Omni::multimodeProcess(const std::string& mode, std::string info) {
    auto file_info = info;

    // 处理图像模式
    if (mode == "img") {
        std::regex hw_regex(R"(<hw>(.*?)</hw>)"); //正则表达式
        std::sregex_iterator iter(info.begin(), info.end(), hw_regex);
        std::sregex_iterator end;
        file_info = "";

        size_t currentPosition = 0;
        if (iter != end) {
            std::smatch match = *iter;
            size_t matchPosition = match.position();
            if (matchPosition > currentPosition) {
                file_info.append(info.substr(currentPosition, matchPosition - currentPosition));
            }

            std::stringstream hw_ss(match.str(1));
            char comma;
            hw_ss >> mVisionHeight >> comma >> mVisionWidth;
            currentPosition = matchPosition + match.length();
        }
        if (currentPosition < info.length()) {
            file_info.append(info.substr(currentPosition));
        }
        // std::cout << "hw: " << mVisionHeight << ", " << mVisionWidth << std::endl;
        // std::cout << "file: " << file_info << std::endl;
    }
    // 若file_info以http开头，则下载文件
    if (file_info.substr(0, 4) == "http") {
        std::regex url_regex(R"(^https?://([^/]+)(/.*))"); //正则表达式
        std::smatch url_match_result;
        std::string host, path;
        if (std::regex_search(file_info, url_match_result, url_regex) && url_match_result.size() == 3) {
            host = url_match_result[1].str();
            path = url_match_result[2].str();
        }
        // std::cout << host << "#" << path << std::endl;
        httplib::Client cli(host);
        auto res  = cli.Get(path);
        file_info = "downloaded_file";
        if (res && res->status == 200) {
            std::ofstream file(file_info, std::ios::binary);
            if (file.is_open()) {
                file.write(res->body.c_str(), res->body.size());
                std::cout << "File has been downloaded successfully." << std::endl;
                file.close();
            } else {
                std::cerr << "Unable to open file to write." << std::endl;
            }
        } else {
            std::cerr << "Failed to download file. Status code: " << (res ? res->status : 0) << std::endl;
        }
    }
    
    // 根据模式及配置选择具体的处理方法
    if (mode == "img" && mConfig->is_visual()) {
        return visionProcess(file_info);
    }
    if (mode == "audio" && mConfig->is_audio()) {
        return audioProcess(file_info);
    }
    return std::vector<int>(0);
}

/**
 * @brief 添加位置ID。位置编码列表用于表示输入的文本或图像的序列位置。
 *
 * 此函数根据传入的参数 t、h 和 w，将位置ID添加到位置ID列表中。
 * 如果 t、h 和 w 的值小于零，则表示添加文本位置ID；否则，添加视觉位置ID。
 *
 * @param t 文本位置ID的起始索引。
 * @param h 视觉位置ID的行数。
 * @param w 视觉位置ID的列数。
 */
void Omni::addPositionIds(int t, int h, int w) {
    int cur_idx = mPositionIds.currentIdx();
    if (h < 0 && w < 0) { // text position ids
        for (int i = 0; i < t; i++) {
            int idx = cur_idx + i;
            mPositionIds.push_back(idx);
        }
    } else { // vision position ids
        // vision start
        mPositionIds.push_back(cur_idx++);
        for (int t_i = 0; t_i < t; t_i++) {
            for (int h_i = 0; h_i < h; h_i++) {
                for (int w_i = 0; w_i < w; w_i++) {
                    mPositionIds.push_back(cur_idx + t_i, cur_idx + h_i, cur_idx + w_i);
                }
            }
        }
        // vision end
        mPositionIds.push_back();
    }
}

/**
 * @brief 对多模态输入进行编码，将文本、图像和音频内容转换为token ID序列
 * 
 * 该函数解析包含<img>和<audio>标签的多模态提示模板，将不同模态的内容分别编码，
 * 并按顺序组合成完整的token ID序列。同时维护位置ID信息。
 * 
 * @param multimodal_input 包含提示模板和相关媒体数据的多模态输入对象
 * @return std::vector<int> 编码后的token ID序列
 */
std::vector<int> Omni::tokenizer_encode(const MultimodalPrompt& multimodal_input) {
    std::string prompt = multimodal_input.prompt_template;
    // MNN_PRINT("tokenizer_encode(MultimodalPrompt) prompt: %s", prompt.c_str());
    
    // 定义匹配<img>或<audio>标签的正则表达式
    std::regex multimode_regex("<(img|audio)>(.*?)</\\1>"); // 多模态正则表达式
    std::string::const_iterator searchStart(prompt.cbegin());
    std::smatch match;
    std::vector<int> ids{};
    mPositionIds.clear();
    
    // 遍历并处理所有匹配的多模态标签
    while (std::regex_search(searchStart, prompt.cend(), match, multimode_regex)) {
        // 编码标签前的文本内容
        auto txt_ids = mTokenizer->encode(match.prefix().str());
        addPositionIds(txt_ids.size());
        ids.insert(ids.end(), txt_ids.begin(), txt_ids.end());

        // 提取标签类型和内容
        std::string mode = match[1].str();
        std::string content = match[2].str();
        std::vector<int> mul_ids;

        // 根据标签类型处理对应(图片或音频)内容
        if (mode == "img") {
            mul_ids = processImageContent(content, multimodal_input.images);
            // MNN_PRINT("tokenizer_encode(MultimodalPrompt) image mul_ids size: %lu", mul_ids.size());
        } else if (mode == "audio") {
            mul_ids = processAudioContent(content, multimodal_input.audios);
            // MNN_PRINT("tokenizer_encode(MultimodalPrompt) audio mul_ids size: %lu", mul_ids.size());
        }
        
        // 将处理后的多模态内容ID添加到结果列表中
        ids.insert(ids.end(), mul_ids.begin(), mul_ids.end());
        searchStart = match.suffix().first;
    }

    // 处理最后剩余的文本内容
    if (searchStart != prompt.cend()) {
        auto txt_ids = mTokenizer->encode(std::string(searchStart, prompt.cend()));
        addPositionIds(txt_ids.size());
        ids.insert(ids.end(), txt_ids.begin(), txt_ids.end());
    }
    return ids;
}



/**
 * @brief 对输入的文本提示进行编码，转换为token序列
 * 
 * 该函数将输入的文本提示转换为多模态提示格式，然后调用重载版本的tokenizer_encode函数
 * 进行实际的编码操作。
 * 
 * @param prompt 输入的文本提示字符串
 * @return std::vector<int> 编码后的token序列，每个token用整数表示
 */
std::vector<int> Omni::tokenizer_encode(const std::string& prompt) {
    MultimodalPrompt multimodal_input;
    multimodal_input.prompt_template = prompt;
    return tokenizer_encode(multimodal_input);
}

/**
 * @brief 处理图像内容，根据输入内容判断是占位符还是文件路径/URL，并进行相应的图像处理
 * @param content 图像内容标识，可以是占位符名称或文件路径/URL
 * @param images 图像占位符映射表，键为占位符名称，值为对应的图像信息
 * @return 返回处理后的图像特征向量
 */
std::vector<int> Omni::processImageContent(const std::string& content, const std::map<std::string, PromptImagePart>& images) {
    
    // 查找内容是否为已知的图像占位符
    auto it = images.find(content);
    if (it != images.end()) {
        // 如果找到占位符且尺寸有效，则更新全局视觉尺寸变量
        if (it->second.height > 0 && it->second.width > 0) {
            mVisionHeight = it->second.height;
            mVisionWidth = it->second.width;
        }
        // MNN_PRINT("processImageContent: using placeholder '%s' with size %dx%d", content.c_str(), mVisionWidth, mVisionHeight);
        return visionProcess(it->second.image_data);
    }
    // MNN_PRINT("processImageContent: treating '%s' as file path or URL", content.c_str());
    // 如果未找到占位符，则将内容视为文件路径或URL进行处理
    return multimodeProcess("img", content);
}


/**
 * @brief 处理音频内容，根据输入内容查找对应的音频数据并进行处理
 * @param content 音频内容标识符，可以是占位符名称或文件路径
 * @param audios 音频映射表，存储占位符名称到PromptAudioPart对象的映射关系
 * @return 处理后的音频数据，以整数向量形式返回；如果处理失败则返回空向量
 */
std::vector<int> Omni::processAudioContent(const std::string& content, const std::map<std::string, PromptAudioPart>& audios) {
    auto it = audios.find(content);
    if (it != audios.end()) {
        // MNN_PRINT("processAudioContent: using placeholder '%s'", content.c_str());
        // 找到了对应的音频占位符，尝试使用waveform或文件路径进行处理
        if (it->second.waveform.get() != nullptr) {
            return audioProcess(it->second.waveform);
        } else if (!it->second.file_path.empty()) {
            return audioProcess(it->second.file_path);
        } else {
            MNN_PRINT("processAudioContent: audio_part has no valid input\n");
            return std::vector<int>(0);
        }
    }
    // MNN_PRINT("processAudioContent: treating '%s' as file path", content.c_str());
    // 未找到对应的占位符，将内容视为文件路径直接处理
    return multimodeProcess("audio", content);
}

/**
 * @brief 对输入的 token ID 序列进行 embedding 处理，支持文本、视觉和音频模态混合输入。
 * 
 * 该函数根据特殊标记（如 mAudioPad、mVisionPad）识别不同模态的输入段，并分别处理：
 * - 文本部分调用基类 Llm::embedding 进行嵌入；
 * - 视觉/音频部分使用预存的 embedding 向量进行替换；
 * 最终将所有 embedding 向量拼接成一个整体输出。
 * 
 * @param input_ids 输入的 token ID 序列，包含文本、视觉和音频标识符。
 * @return VARP 拼接后的 embedding 向量。
 */
VARP Omni::embedding(const std::vector<int>& input_ids) {
    if (input_ids.size() == 1) {
        return Llm::embedding(input_ids);
    }

    // 存储各部分的 embedding 结果
    std::vector<VARP> embeddings;

    std::vector<int> position_ids;

    // 用于记录当前处理到第几个视觉或音频 embedding
    int vision_idx = 0, audio_idx = 0;

    // 当前正在处理的文本 token ID 列表
    std::vector<int> cur_txt_ids;

    // 标志位：是否处于视觉或音频段中
    bool inVision = false, inAudio = false;
    for (int i = 0; i < input_ids.size(); i++) {
        int id = input_ids[i];
        // 音频模块处理逻辑
        if (inAudio) {
            if (id == mAudioPad) {
                continue;
            } else {
                cur_txt_ids.clear();
                inAudio = false;
            }
        } else if (id == mAudioPad) {
            auto txt_embedding = Llm::embedding(cur_txt_ids);
            auto mul_embedding = mAudioEmbeddings[audio_idx++];
            embeddings.push_back(txt_embedding);
            embeddings.push_back(mul_embedding);
            inAudio = true;
        }
        // 视觉模态处理逻辑(两种模式可切换)
#if 1
        if (inVision) {
            if (id == mVisionPad) {
                continue;
            } else {
                cur_txt_ids.clear();
                inVision = false;
            }
        } else if (id == mVisionPad) {
            auto txt_embedding = Llm::embedding(cur_txt_ids);
            auto mul_embedding = mVisionEmbeddings[vision_idx++];
            embeddings.push_back(txt_embedding);
            embeddings.push_back(mul_embedding);
            inVision = true;
        }
        cur_txt_ids.push_back(id);
#else
        if (id == mVisionPad) {
            continue;
        }
        cur_txt_ids.push_back(id);
        if (id == mVisionStart) {
            auto txt_embedding = Llm::embedding(cur_txt_ids);
            auto mul_embedding = mVisionEmbeddings[vision_idx++];
            embeddings.push_back(txt_embedding);
            embeddings.push_back(mul_embedding);
        } else if (id == mVisionEnd) {
            cur_txt_ids.clear();
            cur_txt_ids.push_back(id);
        }
#endif
    }

    mVisionEmbeddings.clear();
    mAudioEmbeddings.clear();


    if (!cur_txt_ids.empty()) {
        auto txt_embedding = Llm::embedding(cur_txt_ids);
        embeddings.push_back(txt_embedding);
    }
    auto embedding = Express::_Concat(embeddings, 0);
    return embedding;
}

static inline bool needNewVar(VARP var, int axis, int seq_len) {
    if (var == nullptr) {
        return true;
    }
    if (var->getInfo()->dim[axis] != seq_len) {
        return true;
    }
    return false;
}

/**
 * @brief 生成位置ID张量，用于模型推理时的位置编码。
 * 
 * 根据序列长度生成对应的位置ID数据。如果模型使用的是常规位置编码，
 * 则调用父类方法生成；如果是mrope（多维度RoPE）方式，则根据不同的
 * 情况生成3个通道的位置ID信息（T、H、W三个维度）。
 *
 * @param seq_len 当前处理的序列长度
 * @return VARP 返回生成的位置ID张量，形状为[3, seq_len]
 */
VARP Omni::gen_position_ids(int seq_len) {
    // 获取模型输入中位置ID的维度信息
    auto positionIdsDims = mModules[0]->getInfo()->inputs[2].dim;

    // 如果batch维度为1，说明是普通的位置编码方式，直接调用父类方法
    if (positionIdsDims[0] == 1) {
        return Llm::gen_position_ids(seq_len);
    }

    // mrope模式：处理多维度RoPE位置编码
    
    // 检查是否需要重新创建位置ID变量（尺寸不匹配时）
    if (needNewVar(positionIds, 1, seq_len)) {
        positionIds = _Input({3, seq_len}, NCHW, halide_type_of<int>());
    }

    // 获取位置ID数据的可写指针
    auto ptr = positionIds->writeMap<int>();

    // 如果是生成模式且已有生成序列长度，则按连续位置生成
    if (mContext->gen_seq_len > 0) {
        for (int i=0; i<seq_len; ++i) {

            // 计算当前位置ID值
            // auto pos = mContext->gen_seq_len + mPositionIds.back() + i;
            auto pos = mContext->all_seq_len + i;

            // 分别设置T、H、W三个维度相同的位置ID
            ptr[i + 0] = pos;
            ptr[i + seq_len] = pos;
            ptr[i + seq_len * 2] = pos;
        }
    } else {
        // 否则按照预定义的位置ID偏移量进行计算T/H/W维度的生成位置ID
        for (int i = 0; i < seq_len; i++) {
            ptr[i] = mPositionIds.mT[i] + mContext->all_seq_len;
            ptr[i + seq_len] = mPositionIds.mH[i] + mContext->all_seq_len;
            ptr[i + seq_len * 2] = mPositionIds.mW[i] + mContext->all_seq_len;
        }
        if (mTalker) { //如果存在对话处理器，则更新其位置ID信息
            mTalker->setPostionIds(mPositionIds);
        }
    }
    // // dump position ids
    // printf("position_ids = [");
    // for (int i = 0; i < seq_len; i++) {
    //     printf("%d ", ptr[i]);
    // }
    // printf("]\n");
    return positionIds;
}


/**
 * @brief 前向传播函数，处理隐藏状态并添加对话嵌入
 * 
 * 该函数首先调用基类Llm的前向传播方法，然后检查是否存在对话处理器，
 * 如果存在且输出结果包含多个元素，则将第二个输出作为对话嵌入添加到对话处理器中。
 * 
 * @param hiddenState 隐藏状态变量，用于前向传播计算
 * @param mask 掩码变量，用于屏蔽无效位置的计算
 * @param inputPos 输入位置变量，指示输入序列中的位置信息
 * @return std::vector<Express::VARP> 前向传播的输出结果向量
 */
std::vector<Express::VARP> Omni::forwardRaw(Express::VARP hiddenState, Express::VARP mask, Express::VARP inputPos) {
    // 调用基类的前向传播方法获取输出
    auto outputs = Llm::forwardRaw(hiddenState, mask, inputPos);
    
    // 如果存在对话处理器且输出包含多个元素，则添加对话嵌入
    if (mTalker && outputs.size() > 1) {
        mTalker->addTalkerEmbeds(outputs[1]);
    }
    return outputs;
}

/**
 * @brief 响应函数，根据输入ID生成文本输出
 * 
 * 该函数初始化生成器并调用generate函数来生成新的文本序列。
 * 
 * @param input_ids 输入的token ID序列
 * @param os 输出流指针，用于输出生成的文本
 * @param end_with 结束标记字符串，默认为换行符
 * @param max_new_tokens 最大新生成的token数量
 */
void Omni::response(const std::vector<int>& input_ids, std::ostream* os, const char* end_with, int max_new_tokens) {
    // 设置默认结束标记
    if (!end_with) { end_with = "\n"; }

    // 初始化文本生成器
    generate_init(os, end_with);
    
    // 如果存在对话处理器，则初始化对话处理器的生成器
    if (mTalker) {
        mTalker->generate_init();
    }

    // 执行文本生成过程
    generate(input_ids, max_new_tokens);
}


/**
 * @brief 设置音频波形数据回调函数
 * 
 * 该函数用于设置处理音频波形数据的回调函数。当有音频数据需要处理时，
 * 系统会调用此回调函数。回调函数接收音频数据指针、数据长度和结束标志。
 * 
 * @param callback 回调函数对象，参数说明：
 *                 - const float*: 音频波形数据指针
 *                 - size_t: 音频数据长度
 *                 - bool: 是否为最后一块数据的标志
 *                 回调函数返回bool值表示处理是否成功
 */
void Omni::setWavformCallback(std::function<bool(const float*, size_t, bool)> callback) {
    if (mTalker) {
        mTalker->setWavformCallback(callback);
    }
}


/**
 * @brief 生成语音波形数据
 * 
 * 该函数调用 Talker 对象的 generate 方法来生成语音波形。如果定义了性能统计宏，
 * 还会输出详细的性能指标信息，包括各个阶段的耗时、处理速度以及实时因子等。
 * 
 * @note 该函数不接受参数，无返回值
 */
void Omni::generateWavform() {
    if (mTalker) {
        mTalker->generate();
#ifdef DUMP_TALKER_PERFORMANCE
        // 获取上下文信息用于性能统计
        auto context = mTalker->getContext();

        // 将微秒转换成秒
        float prefill_s = context->prefill_us / 1e6;
        float decode_s = context->decode_us / 1e6;
        float token2wav_s = context->audio_us / 1e6;
        float dit_s = context->vision_us / 1e6;
        
        // 计算文本到语音（TTS）总时间
        float tts_s = token2wav_s;        
        if (mTalker->mStreamWithDecode) {
            tts_s += decode_s;
        }

        // 根据生成序列长度计算音频时长（假设采样率为50Hz）
        float audio_duration = context->gen_seq_len / 50.0;
        printf("\n#################################\n");
        printf("prompt tokens num = %d\n", context->prompt_len);
        printf("decode tokens num = %d\n", context->gen_seq_len);
        printf("  prefill time = %.2f s\n", prefill_s);
        printf("   decode time = %.2f s\n", decode_s);
        printf("      dit time = %.2f s\n", dit_s);
        printf("token2wav time = %.2f s\n", token2wav_s);
        printf("      tts time = %.2f s\n", tts_s);
        printf("  prefill speed = %.2f tok/s\n", context->prompt_len / prefill_s);
        printf("   decode speed = %.2f tok/s\n", context->gen_seq_len / decode_s);
        printf("token2wav speed = %.2f tok/s\n", context->gen_seq_len / token2wav_s);
        printf("      tts rtf   = %.2f \n", tts_s / audio_duration);
        printf("##################################\n");
#endif
    }
}


/**
 * @brief 初始化 Talker 模块，加载模型、配置参数和相关嵌入表示。
 *
 * 该函数完成以下主要工作：
 * 1. 初始化运行时环境；
 * 2. 设置采样器配置并创建采样器；
 * 3. 加载磁盘上的 Embedding 数据；
 * 4. 加载说话人相关参数（如 spk、cond、BOS/EOS/PAD token）；
 * 5. 加载各个推理模块（包括 Talker 主模型、DiT 模型、BigVGAN 模型等）；
 * 6. 克隆模块用于自回归解码和预填充。
 */
void Talker::load() {
    // 初始化运行时环境
    initRuntime();

    // 设置序列长度索引为 1
    mSeqLenIndex = 1;

    // 设置采样器相关配置参数
    set_config("{\"sampler_type\": \"mixed\", \"temperature\": 0.9, \"topK\": 40, \"topP\": 0.8, \"penalty\": 1.05}");
    
    // 创建采样器实例
    mSampler.reset(Sampler::createSampler(mContext, mConfig));
    
    // 加载磁盘上的 Embedding 数据
    mDiskEmbedding.reset(new DiskEmbedding(mConfig, mConfig->talker_embedding_file()));
    
    // 获取最大新生成token数
    mMaxNewTokens = mConfig->talker_max_new_tokens();
    
    // 获取当前说话人名称，并加载对应的说话人字典
    std::string speaker = mConfig->talker_speaker();
    auto spk_dict = Express::Variable::loadMap(mConfig->spk_dict().c_str());

    // 提取说话人相关嵌入向量和特殊 token
    mSpk = spk_dict[speaker + "_spk"];
    mCond = spk_dict[speaker + "_cond"];
    mTextBosToken = int(spk_dict[speaker + "_bos_token"]->readMap<float>()[0]);

    // 构造文本和 codec 的 BOS/EOS/PAD 嵌入向量
    mTextBos = mThinker->embedding({mTextBosToken});
    mTextEos = mThinker->embedding({mTextEosToken});
    mTextPad = mThinker->embedding({mTextPadToken});
    mCodecBos = embedding({mCodecBosToken});
    mCodecPad = embedding({mCodecPadToken});

    // 配置推理模块参数
    Module::Config module_config;
    module_config.shapeMutable = false;
    module_config.rearrange    = true;

    // 调整模块容器大小
    mModules.resize(1);

    // 定义输入名称列表
    std::vector<std::string> inputNames {"inputs_embeds", "attention_mask", "position_ids", "logits_index"};

    // 加载主 Talker 模型
    mModules[0].reset(Module::load(inputNames,
                                    {"logits"}, mConfig->talker_model().c_str(), mRuntimeManager, &module_config));
    
    // 加载预处理 DiT 模型（条件、说话人、编码输入）
    mPreDit.reset(Module::load({"cond", "spk", "code"}, {"code_embeds", "rope", "mask"},
                                mConfig->predit_model().c_str(), mRuntimeManager, &module_config));
    
    // 加载 DiT 模型（用于生成mel频谱）
    mDit.reset(Module::load({"x", "code_embeds", "rope", "mask", "time"}, {"mel"},
                            mConfig->dit_model().c_str(), mRuntimeManager, &module_config));
    
    // 加载 BigVGAN 模型（将 mel 频谱转换为波形）
    mBigvgan.reset(Module::load({"generated_mel"},
                                {"waveform"}, mConfig->bigvgan_model().c_str(), mRuntimeManager, &module_config));
    
    // 克隆模块用于自回归解码
    mModulePool[std::make_pair(1, false)].reset(Module::clone(mModules[0].get()));
    
    // 将主模块作为预填充模块存入池中
    mModulePool[std::make_pair(mPrefillKey, mConfig->all_logits())] = mModules[0];
}


/**
 * @brief 初始化生成器状态，准备进行语音合成
 * 
 * 此函数负责初始化Talker对象的内部状态，包括清理嵌入向量、
 * 生成初始噪声数据、预留波形缓冲区等操作，为后续的语音生成做准备。
 * 
 * @param os 输出流指针，用于输出初始化相关信息
 * @param end_with 字符串指针，指定输出结束标记
 */
void Talker::generate_init(std::ostream* os, const char* end_with) {
    if (!doGenerate()) { return; }
    Llm::generate_init(os, end_with);
    
    // 初始化流式生成相关状态
    mTalkerEmbeds.clear();

    // 如果初始噪声为空，则生成符合正态分布的随机噪声数据
    if (mInitialNoise.empty()) {
        mInitialNoise.resize(mMaxNewTokens * 2 * 80);
        std::random_device rd;
        std::mt19937 generator(rd());
        std::normal_distribution<double> distribution(0.0, 1.0);
        for (int i = 0; i < mMaxNewTokens * 2 * 80; ++i) {
            mInitialNoise[i] = distribution(generator);
        }
    }

    // 预留波形缓冲区内存空间
    mWaveformBuffer.reserve(mMaxNewTokens * 2 * 240);

    // 重置相关缓冲区和索引变量
    mMelBuffer = nullptr;
    dit_start_index = 0;
    dit_left_padding = 0;
    vocoder_left_pad = 0;
}

/**
 * @brief 获取输入token序列的词嵌入表示
 * 
 * 该函数将输入的token ID序列转换为对应的词嵌入向量表示，
 * 用于后续的神经网络计算。
 * 
 * @param input_ids 输入的token ID序列，每个ID对应词汇表中的一个词
 * @return 返回词嵌入后的张量表示，维度通常为[sequence_length, embedding_dim]
 */
Express::VARP Talker::embedding(const std::vector<int>& input_ids) {
    return Llm::embedding(input_ids);
}

/**
 * @brief 生成位置ID张量，用于多专家机制(mrope)位置编码
 * @param seq_len 序列长度
 * @return 包含位置ID信息的VARP变量
 */
Express::VARP Talker::gen_position_ids(int seq_len) {
    // mrope
    if (needNewVar(positionIds, 2, seq_len)) {
        positionIds = _Input({3, 1, seq_len}, NCHW, halide_type_of<int>());
    }
    auto ptr = positionIds->writeMap<int>();
    if (seq_len == 1) {
        ptr[0] = mContext->gen_seq_len + mPositionIds.back();
        ptr[1] = ptr[0];
        ptr[2] = ptr[0];
    } else {
        for (int i = 0; i < seq_len; i++) {
            ptr[i] = mPositionIds.mT[i];
            ptr[i + seq_len] = mPositionIds.mH[i];
            ptr[i + seq_len * 2] = mPositionIds.mW[i];
        }
    }
    return positionIds;
}

void Talker::setWavformCallback(const std::function<bool(const float*, size_t, bool)> callback) {
    mWavformCallback = callback;
}

/**
 * @brief 执行 DiT（Diffusion Transformer）模型的前向推理过程，用于生成语音合成中的梅尔频谱图。
 *
 * 此函数使用预训练的 DiT 模型进行扩散过程的逆推演，从噪声逐步生成目标音频特征。
 *
 * @param codec_size 编码 token 的数量。
 * @param codec_tokens 指向编码 token 序列的指针。
 * @param initial_noise 初始噪声数据指针，若为空则自动生成标准正态分布噪声。
 * @return 返回生成的梅尔频谱图张量（维度为 [1, 80, max_duration]）。
 */
VARP Talker::ditForward(const int codec_size, const int* codec_tokens, const float* initial_noise) {
    // 将输入的 codec tokens 转换为表达式变量
    auto code = _Const(codec_tokens, {1, codec_size}, NCHW, halide_type_of<int>());
    
    // 计算最大持续时间：token 数量的两倍
    const int max_duration = codec_size * 2;

    // 前向传播获取条件嵌入、位置编码和掩码等中间结果
    auto outputs = mPreDit->onForward({mCond, mSpk, code});
    auto code_embeds = outputs[0];  // 条件嵌入
    auto rope = outputs[1];         // RoPE 位置编码
    auto mask = outputs[2];         // 注意力掩码

    // 获取配置中指定的扩散步数与求解器类型
    const int steps = mConfig->dit_steps();
    const int solver = mConfig->dit_solver();

    // 定义时间步长比例因子
    const float step_ratio = 1.0 / (steps - 1);

    // 定义一个 lambda 函数封装 DiT 模型的一次前向调用
    auto forward_dit = [&](float t, Express::VARP x) {
        auto pred = mDit->onForward({x, code_embeds, rope, mask, _Const(t, {1}, NCHW)})[0];
        return pred;
    };

    // 初始化输出变量 y0，形状为 [1, max_duration, 80]
    auto y0 = _Input({1, max_duration, 80}, NCHW, halide_type_of<float>());
    
    // 根据是否提供初始噪声决定初始化方式
    if (initial_noise) {
        for (int i = 0; i < max_duration * 80; ++i) {
            y0->writeMap<float>()[i] = initial_noise[i];
        }
    } else { // 若未提供初始噪声，则随机生成符合标准正态分布的噪声
        std::random_device rd;
        std::mt19937 generator(rd());
        std::normal_distribution<double> distribution(0.0, 1.0);
        for (int i = 0; i < max_duration * 80; ++i) {
            y0->writeMap<float>()[i] = distribution(generator);
        }
    }

    // 开始计时统计
    MNN::Timer _t;

    // 进行多步迭代反向扩散过程
    for (int i = 0; i < steps - 1; i++) {
        // 使用余弦调度计算当前步的时间点 t0 和下一步 t1
        float t0 = 1 - std::cos(M_PI / 2 * i * step_ratio);
        float t1 = 1 - std::cos(M_PI / 2 * (i + 1) * step_ratio);
        float dt = t1 - t0;

        // 第一阶段： 计算 k1（欧拉法或 RK4 法的第一项）
        auto k1 = mDit->onForward({y0, code_embeds, rope, mask, _Const(t0, {1}, NCHW)})[0];
        if (solver == 1) {
            // 使用简单欧拉方法更新 y0
            y0 = y0 + k1 * _Scalar<float>(dt);
        } else {
            // 使用四阶Runge-Kutta方法进行更精确的数值积分
            constexpr float one_third = 1.0 / 3.0;
            constexpr float two_third = 2.0 / 3.0;
            auto kk1 = _Clone(k1, true); // 备份，避免重复计算
            auto k2 = forward_dit(t0 + dt * one_third, y0 + k1 * _Scalar<float>(dt * one_third));
            auto kk2 = _Clone(k2, true);
            auto k3 = forward_dit(t0 + dt * two_third, y0 + _Scalar<float>(dt) * (k2 - k1 * _Scalar<float>(two_third)));
            auto kk3 = _Clone(k3, true);
            auto k4 = forward_dit(t1, y0 + _Scalar<float>(dt) * (k1 - k2 + k3));
            auto kk4 = _Clone(k4, true);

            // 组合所有系数得到最终增量，并更新 y0
            auto dy = (kk1 + _Scalar<float>(3.0) * (kk2 + kk3) + kk4) * _Scalar<float>(dt * 0.125);
            y0 = y0 + dy;
        }
    }

    // 累加本次操作所花费的时间到上下文计时器中
    mContext->vision_us += _t.durationInUs();

    // 对输出张量做转置，将通道维度调整至第二位
    auto generated_mel = _Permute(y0, {0, 2, 1});
    return generated_mel;
}

/**
 * @brief 使用BigVGAN模型将梅尔频谱图转换为波形音频
 * 
 * @param mel 输入的梅尔频谱图张量
 * @return VARP 生成的音频波形张量
 */
VARP Talker::bigvganForward(VARP mel) {
    // 调用BigVGAN模型进行前向推理，将梅尔频谱图转换为音频波形
    auto waveform = mBigvgan->forward(mel);
    return waveform;
}

/**
 * @brief 将生成的 codec token 转换为音频波形数据（WAV），并进行流式处理。
 *
 * 该函数负责将模型输出的 token 序列分块转换为对应的 Mel 频谱图，再通过声码器（BigVGAN）生成波形数据。
 * 支持流式处理，每次处理一个 chunk，并维护内部状态以支持连续调用。
 *
 * @param talker_done 表示是否已经完成 token 生成。若为 true，则表示这是最后一次调用。
 */
void Talker::token2wav(bool talker_done) {
    // 计算当前剩余需要处理的 codec token 数量
    int codec_size = mContext->gen_seq_len - dit_start_index;

    // 当前 chunk 的大小，包含左、中、右 padding
    int chunk_size = dit_left_padding + dit_chunk_size + dit_right_padding;

    // 判断当前是否是最后一个 chunk：token 已全部生成且剩余 token 不超过 chunk 大小
    bool last_chunk = talker_done && (codec_size <= chunk_size);
    // prefill some codec tokens
    // if (!talker_done && mMelBuffer == nullptr && codec_size < chunk_size * 2) {
    //     return;
    // }

    // 如果不是最后一个 chunk，但剩余 token 数不足一个 chunk，则暂不处理
    if (!last_chunk && codec_size < chunk_size) {
        return;
    }

    // 获取当前要处理的 codec token 指针和对应的初始噪声数据指针
    auto codec_ptr = mContext->output_tokens.data() + dit_start_index;
    auto noise_ptr = mInitialNoise.data() + dit_start_index * 160;

    // 确定本次实际处理的 token 数量
    int real_size = last_chunk ? codec_size : chunk_size;

    // 确定本次生成的 Mel 频谱图中有效部分的大小
    int mel_size = last_chunk ? -1 : dit_chunk_size * 2;
    MNN::Timer _t;
    
    // 使用 DIT 模型将 token 转换为 Mel 频谱图
    auto generated_mel = ditForward(real_size, codec_ptr, noise_ptr);

    // 对生成的 Mel 频谱图裁剪，去除 padding 部分
    generated_mel = _Slice(generated_mel, _var<int>({0, 0, dit_left_padding * 2}, {3}), _var<int>({-1, -1, mel_size}, {3}));
    
    // 更新 Mel buffer：首次处理则直接赋值，否则拼接
    mMelBuffer = (mMelBuffer == nullptr) ? generated_mel : _Concat({mMelBuffer, generated_mel}, -1);
    
    // 更新下次处理的左 padding 为上下文长度
    dit_left_padding = dit_left_context;

    // 更新下次处理的起始索引，跳过已处理的部分（减去重叠区域）
    dit_start_index += (chunk_size - dit_left_padding - dit_right_padding);
    
    // 使用 BigVGAN 声码器将 Mel 频谱图转换为波形数据
    auto generated_waveform = bigvganForward(mMelBuffer);

    // 获取波形数据的有效部分（去除 padding）
    auto ptr = generated_waveform->readMap<float>() + vocoder_left_pad * vocoder_upsample_rate;
    auto size = generated_waveform->getInfo()->size - (vocoder_left_pad + vocoder_right_pad) * vocoder_upsample_rate;
    
    // 将生成的波形数据追加到总波形缓冲区中
    mWaveformBuffer.insert(mWaveformBuffer.end(), ptr, ptr + size);

    // 更新下次 vocoder 的左 padding 为上下文长度
    vocoder_left_pad = vocoder_left_context;

    // 更新 Mel buffer，保留用于下次处理的上下文部分
    mMelBuffer = _Slice(mMelBuffer, _var<int>({0, 0, -vocoder_left_pad - vocoder_right_pad}, {3}), _var<int>({-1, -1, -1}, {3}));
    
    // 累计音频处理耗时
    mContext->audio_us += _t.durationInUs();

    // 如果设置了波形回调函数，则调用它处理当前生成的波形数据
    if (mWavformCallback) {
        bool res = mWavformCallback(ptr, size, last_chunk);
        if (!res) { return; }
    }

    // 如果 token 已全部生成但不是最后一个 chunk（即还有剩余 token），递归调用继续处理
    if (talker_done && !last_chunk) {
        token2wav(true);
    }
}

/**
 * @brief 将编码器tokens转换为音频波形数据
 * 
 * 该函数通过两阶段处理将编码器tokens转换为最终的音频波形：
 * 1. 使用DiT模型将编码器tokens前向传播生成梅尔频谱图
 * 2. 使用BigVGAN模型将梅尔频谱图转换为音频波形
 * 
 * @param codec_tokens 编码器tokens序列，包含音频的离散表示
 * @return VARP 处理后的音频波形数据张量
 */
VARP Talker::token2wav(const std::vector<int>& codec_tokens) {
    auto generated_mel = ditForward(codec_tokens.size(), codec_tokens.data());
    auto waveform = bigvganForward(generated_mel);
    return waveform;
}

/**
 * @brief 对给定的logits进行采样，获取下一个token，并根据需要执行音频解码
 * 
 * 该函数首先调用Llm::sample对输入的logits张量进行采样操作，
 * 获取下一个token。如果启用了流式解码模式(mStreamWithDecode为true)，
 * 则会调用token2wav()函数将token转换为音频数据。
 * 
 * @param logits 包含词汇表概率分布的张量变量
 * @param offset 采样操作的起始偏移位置
 * @param size 采样操作的有效大小范围
 * @return 采样得到的token索引值
 */
int Talker::sample(Express::VARP logits, int offset, int size) {
    // 调用Llm类的静态采样函数获取下一个token
    int token = Llm::sample(logits, offset, size);

    // 如果启用了流式解码模式，则调用token2wav()函数将token转换为音频数据
    if (mStreamWithDecode) {
        token2wav();
    }
    return token;
}


/**
 * @brief 执行文本到语音的生成过程
 * 
 * 该函数负责控制整个语音合成流程，包括预处理输入、执行模型推理、采样输出token以及后处理生成音频。
 * 函数首先检查是否需要进行生成，然后构建初始输入嵌入向量，接着通过模型前向传播生成第一个token，
 * 最后循环生成后续token直到达到最大长度或遇到结束标记，并将最终结果转换为音频波形。
 */
void Talker::generate() {
    // 检查是否需要执行生成过程，如不需要则直接返回
    if (!doGenerate()) { return; }

    // 在说话人嵌入序列末尾添加文本结束标记
    mTalkerEmbeds.push_back(mTextEos);

    // 构建模型输入嵌入向量：拼接首个说话人嵌入、文本开始标记与编解码器填充、第二个说话人嵌入与编解码器开始标记
    auto input_embeds = _Concat({mTalkerEmbeds[0], mTextBos + mCodecPad, mTalkerEmbeds[1] + mCodecBos}, 1);
    
    // 添加两个位置ID占位符（实际使用中可能在其他地方赋值）
    mPositionIds.push_back();
    mPositionIds.push_back();

    // 记录提示序列长度用于上下文管理
    mContext->prompt_len = input_embeds->getInfo()->dim[1];

    // 启动计时器以测量预填充阶段耗时
    MNN::Timer _t;

    // 执行首次前向传播获取logits输出
    auto logits = forward(input_embeds);

    // 对logits进行采样获得当前token，并将其加入历史和输出token列表
    mContext->current_token = sample(logits);
    mContext->history_tokens.push_back(mContext->current_token);
    mContext->output_tokens.push_back(mContext->current_token);

    // 累加预填充阶段所用时间
    mContext->prefill_us += _t.durationInUs();
    _t.reset(); // 重置计时器以测量解码阶段耗时

    // 循环生成新的token，最多生成mMaxNewTokens个
    for (int i = 1; i < mMaxNewTokens; i++) {
        input_embeds = embedding({mContext->current_token});

        // 根据当前步骤决定是否添加对应的说话人嵌入或默认填充
        if (i + 1 < mTalkerEmbeds.size()) {
            input_embeds = input_embeds + mTalkerEmbeds[i + 1];
        } else {
            mTalkerEmbeds.clear();
            input_embeds = input_embeds + mTextPad;
        }

        auto logits = forward(input_embeds);
        mContext->current_token = sample(logits);
        mContext->history_tokens.push_back(mContext->current_token);
        mContext->output_tokens.push_back(mContext->current_token);

        // 判断是否到达特殊结束标记(8292 或 8294)，若是则提前终止生成
        if (mContext->current_token == 8292 || mContext->current_token == 8294) {
            break;
        }
    }

    // 累加解码阶段所用时间
    mContext->decode_us += _t.durationInUs();

    // 将生成的token序列转换为音频波形数据
    token2wav(true);
}

void Talker::setPostionIds(const MropeInfo& positionIds) {
    // 检查是否允许生成，如果不允许则直接返回
    if (!doGenerate()) { return; }
    // 拷贝赋值位置ID信息
    mPositionIds = MropeInfo(positionIds);
}

void Talker::addTalkerEmbeds(VARP talker_embeds) {
    // 检查是否允许生成，如果不允许则直接返回
    if (!doGenerate()) { return; }
    // 将输入的嵌入表示克隆后添加到列表中
    mTalkerEmbeds.push_back(_Clone(talker_embeds, true));
}

} // namespace Transformer
} // namespace MNN
