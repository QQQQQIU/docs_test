# 概况

Tecorigin Deep Neural Network library
(tecoDNN)是一个面向SW-AI芯片的深度学习加速库。tecoDNN提供了一系列的高性能算子实现，用于支撑深度学习模型的训练和推理过程。
数据类型参考文档提供了所有数据类型和枚举变量的说明，API参考文档则详细描述了各个算子接口的具体参数和使用方法。

## 版本历史

| 文档名称 | TecoDNN 接口手册                     |
|----------|--------------------------------------|
| 版本     | V1.4.4                               |
| 作者     | High Performance Engineering （HPE） |
| 时间     | 2022.09.05                      |

<a href="../_images/tecodnn-v1.4.2.pdf" target="_blank">离线版本文档</a>


## 更新记录

### v1.4.5更新记录
- 更改CELoss，BCELoss接口
- 新增MSELossF的算子

### v1.4.4更新记录

- 优化部分convF实现
- 优化nms的实现

- 解除rnnBD,rnnBW对half类型的限制条件
- 解除MHAttnF, MHAttnBD,MHAttnBW的half类型限制
- 解除poolingF，poolingB的half类型限制
- 解除actF，actB对half类型的限制
- 解除addTensor, scaleTensor, setTensor对half类型的限制

- 新增swdnnAdaptivePooling前向和反向算子, 支持float和half(存在限制)
- 新增swdnnGroupNorm前向和反向算子, 支持float和half(存在限制)
- 新增swdnnCELoss前向和反向算子, 支持float和half(存在限制)
- 新增swdnnBCELoss前向算子（反向尚未加入）, 支持float和half(存在限制)
- 新增swdnnCornerPool前向和反向算子, 支持float和half(存在限制)
- 新增swdnnAny算子, 支持float和half(存在限制)
- 新增swdnnAll算子, 支持float和half(存在限制)
- 新增swdnnBernoulli算子, 支持float和half(存在限制)
- 新增swdnnConstant算子, 支持float和half(存在限制)
- 新增swdnnEyeLike算子, 支持float和half(存在限制)
- 新增swdnnBitShift算子, 支持float和half(存在限制)

### v1.4.3更新记录

- 修改swdnnInplaceMode_t的值
- 修改swdnnSoftplusBackward的接口原型
- 修改swdnnSoftsignBackward的接口原型

- swdnnActivationMode_t新增4个模式
	- SWDNN_ACTIVATION_LEAKYRELU
	- SWDNN_ACTIVATION_SELU
	- SWDNN_ACTIVATION_RELU6
	- SWDNN_ACTIVATION_SILU
- swdnnFusedOps_t新增SWDNN_FUSED_CONV_BIAS_ACTIVATION_FORWARD模式, 支持half类型
- 增加swdnnConvolutionBatchNormActivationForwardInference算子, 支持half,float类型
- 增加swdnnConcat算子, 支持half，float类型
- 增加swdnnSplit算子, 支持half，float类型
- 增加swdnnAbsGrad算子, 支持half，float类型
- 增加swdnnInplaceOps算子, 支持half，float类型
- 增加swdnnCumsum算子, 支持half，float类型
- 增加swdnnNms算子, 支持float类型
- 优化swdnnConvolutionBackwardData的实现

### v1.4.2更新记录

- 增加ConvolutionForward接口功能
	- 增加group convolution, 支持half, float
	- 增加depthwise convolution, 支持half, float
	- 增加3D卷积，支持half（暂不支持float) 

- 增加ConvolutionBackwardData接口功能
	- 增加group convolution, 支持half, float
	- 增加depthwise convolution, 支持half, float
	- 增加3D卷积，支持half（暂不支持float) 

- 增加ConvolutionBackwardFilter接口功能
	- 增加group convolution, 支持half, float
	- 增加depthwise convolution, 支持half, float
	- 增加3D卷积，支持half（暂不支持float) 

- 新增swdnnUnaryOps，支持log/exp/sqrt等共19种运算和数据转换，支持half，float；

- 新增Activation GELU模式，支持half，float；

- 新增LogSoftmax模式，支持half，float;

- 新增swdnnClampTensor算子，支持half，float;

- 新增swdnnReciprocalTensor算子，支持half，float；

- 新增swdnnTopk算子，支持half，float;

###  v1.4.1更新记录

-   新增swdnnAddTensorEx算子，支持half,float, 支持双向广播;

-   新增swdnnSubTensorEx算子，支持half,float, 支持双向广播;

-   新增swdnnMulTensorEx算子，支持half,float, 支持双向广播;

-   新增swdnnDivTensorEx算子，支持half,float, 支持双向广播;

-   新增swdnnSquaredDifference算子，支持half,float, 支持双向广播;

-   新增swdnnTensorEqual算子，支持int,half,float,short, 支持双向广播;

-   新增swdnnTensorGreater算子，支持int,half,float,short, 支持双向广播;

-   新增swdnnTensorLess算子，支持int,half,float,short, 支持双向广播;

-   新增swdnnBitwiseAndTensor算子，支持int, 支持双向广播;

-   新增swdnnBitwiseNotTensor算子，支持int, 支持双向广播;

-   新增swdnnBitwiseOrTensor算子，支持int, 支持双向广播;

-   新增swdnnBitwiseXorTensor算子，支持int, 支持双向广播;

-   新增swdnnLogicalAndTensor算子，支持int, 支持双向广播;

-   新增swdnnLogicalNotTensor算子，支持int, 支持双向广播;

-   新增swdnnLogicalOrTensor算子，支持int, 支持双向广播;

-   新增swdnnLogicalXorTensor算子，支持int, 支持双向广播;

-   新增swdnnWhereTensor算子，支持half,float;

-   新增swdnnNegTensor算子，支持half,float;

-   新增swdnnSoftplusForward算子，支持half,float;

-   新增swdnnSoftplusBackward算子，支持half,float;

-   新增swdnnSoftsignForward算子，支持half,float;

-   新增swdnnSoftsignBackward算子，支持half,float;

-	新增swdnnRandomUniform算子，支持half, float

-   新增swdnnRoiPoolingForward算子，支持half,float;

-   新增swdnnRoiPoolingBackward算子，支持half,float;

-   新增swdnnRoiAlignForward算子，支持half,float；

-   新增swdnnRoiAlignBackward算子，支持half,float;

-	新增swdnnTruncatedNormal算子，支持half,float

###  v1.4.0更新记录

-   新增swdnnOpTensor算子

-   新增swdnnReduceTensor算子

-   新增swdnnClipTensor算子

-   新增swdnnFusedExecute算子

-   新增swdnnGumbelSoftmax算子

-   新增所有算子对half类型的支持（除RNNFI和RNNFT）

# 通用描述

## 编程模型

tecoDNN库根据芯片的结构和功能特点，支持独立（RC）模式和加速卡（EP）模式。

tecoDNN库向独立模式的用户提供一套主核API接口，每个应用通过调用swdnnCreate()初始化一个句柄，通过主核运行时获取单/多个核组的资源，仅支持单进程情况下，这个句柄作为参数，传递到随后调用每一个库函数中，最后通过swdnnDestroy()释放核组资源。单核组的独立模式的内存模型如下图所示。主核和从核共享统一的共享内存视图，因此独立模式下，主核和从核可以通过主存进行数据共享、同步、以及参数传递等操作。

![image-image1](media/image1.png)  

tecoDNN库向加速卡模式的用户提供一套HOST端API接口，计算需要访问的数据必须存放于sw-AI芯片端。每个应用通过调用swdnnCreate()初始化一个句柄，通过sdaa运行时获取芯片内单/多个核组的资源，仅支持单进程情况下，这个句柄作为参数，传递到随后调用每一个库函数中，最后通过swdnnDestroy()释放芯片和核组资源。单核组的加速卡模式内存模型如下图所示，host的主存与device的主存独立分开，数据交互需要经过PCIe。host代码通过sddaMalloc/sdaaFree等运行时接口管理device的内存空间，通过sdaaMemcpy进行host-device的数据传输。因此，需要特别注意API接口中的指针参数，确定其所指向的存储具体是host还是device的内存空间。

![image-image2](media/image2.png)  

## 张量描述符

四维张量的存储格式为NCHW、NHWC和CHWN。计算函数默认支持NHWC，为了充分发挥芯片的性能并获得最新的接口支持，推荐用户使用NHWC存储格式。

# 数据类型参考

用于描述tecoDNN库API中使用到的所有结构体类型和枚举变量。

## swdnnActivationDescriptor_t

swdnnActivationDescriptor_t是个结构体指针，通过swdnnCreateActivationDescriptor()创建一个描述符，通过swdnnSetActivationDescriptor()初始化这个描述符。通过swdnnGetActivationDescriptor()来获取这个描述符，并通过swdnnDestoryActivationDescriptor()来销毁这个描述符。

## swdnnActivationMode_t

swdnnActivationMode_t是枚举变量，用于选择神经元激活函数。

1. SWDNN_ACTIVATION_SIGMOID
2. SWDNN_ACTIVATION_RELU
3. SWDNN_ACTIVATION_TANH
4. SWDNN_ACTIVATION_CLIPPED_RELU
5. SWDNN_ACTIVATION_ELU
6. SWDNN_ACTIVATION_IDENTITY（暂不支持）
7. SWDNN_ACTIVATION_SIGMOID_TAB（高性能版本，但是精度较标准版低）
8. SWDNN_ACTIVATION_ELU_TAB（高性能版本，但是精度较标准版低）
9. SWDNN_ACTIVATION_TANH_TAB（高性能版本，但是精度较标准版低）
10. SWDNN_ACTIVATION_GELU
11. SWDNN_ACTIVATION_LEAKYRELU
12. SWDNN_ACTIVATION_SELU
13. SWDNN_ACTIVATION_RELU6
14. SWDNN_ACTIVATION_SILU

## swdnnAttnDescriptor_t

swdnnAttnDescriptor_t是一个结构体指针，通过swdnnCreateAttnDescriptor()
创建一个描述符，通过swdnnSetAttnDescriptor()初始化这个描述符。通过swdnnGetAttnDescriptor()来查询之前初始化的描述符，通过swdnnDestoryAttnDescriptor()来销毁这个描述符。

其中attnMode支持SWDNN_ATTN_QUERYMAP_ALL_TO_ONE、SWDNN_ATTN_QUERYMAP_ONE_TO_ONE、SWDNN_ATTN_DISABLE_PROJ_BIASES、SWDNN_ATTN_ENABLE_PROJ_BIASES（暂不支持）

## swdnnBatchNormMode_t

swdnnBatchNormMode_t是枚举变量，用于选择BatchNormalization操作模式。

1. SWDNN_BATCHNORM_PER_ACTIVATION 
2. SWDNN_BATCHNORM_SPATIAL 
3. SWDNN_BATCHNORM_SPATIAL_PERSISTENT（暂不支持）

注：
SWDNN_BATCHNORM_PER_ACTIVATION：bnBias和bnScale 的维度是1xCxHxW。
SWDNN_BATCHNORM_SPATIAL：bnBias和bnScale 的维度是1xCx1x1。

## swdnnConvolutionBwdDataAlgo_t

swdnnConvolutionBwdDataAlgo_t是枚举变量，用于选择数据反向卷积的算法。

1. SWDNN_CONVOLUTION_BWD_DATA_ALGO_0

## swdnnConvolutionBwdFilterAlgo_t

swdnnConvolutionBwdFilterAlgo_t是枚举变量，用于选择卷积核反向卷积的算法。

1. SWDNN_CONVOLUTION_BWD_FILTER_ALGO_0

2. SWDNN_CONVOLUTION_BWD_FILTER_ALGO_1（暂不支持）

## swdnnConvolutionDescriptor_t

swdnnConvolutionDescriptor_t是结构体指针，通过swdnnCreateConvolutionDescriptor()
创建描述符，通过swdnnSetConvolutionNdDescriptor()
或者swdnnSetConvolution2dDescriptor()
初始化由swdnnCreateConvolutionDescriptor()创建的描述符。通过swdnnGetConvolutionNdDescriptor()或者swdnnGetConvolution2dDescriptor()获取描述符，通过swdnnDestoryConvolutionDescriptor()销毁描述符。

## swdnnConvolutionFwdAlgo_t

swdnnConvolutionFwdAlgo_t是枚举变量，用于选择正向卷积的算法。

1. SWDNN_CONVOLUTION_FWD_ALGO_0

2. SWDNN_CONVOLUTION_FWD_ALGO_ACE_1（暂不支持）

3. SWDNN_CONVOLUTION_FWD_ALGO_ACE_2（暂不支持）

## swdnnConvolutionMode_t

swdnnConvolutionMode_t用于选择卷积操作模式的枚举变量。

1. SWDNN_CONVOLUTION（暂不支持）

2. SWDNN_CROSS_CORRELATION

## swdnnDataType_t

swdnnDataType_t是数据类型。

1. SWDNN_DATA_FLOAT32

2. SWDNN_DATA_HALF

3. SWDNN_DATA_INT8

4. SWDNN_DATA_INT16

5. SWDNN_DATA_INT32

## swdnnDirectionMode_t

swdnnDirectionMode_是枚举变量，在swdnnRNNForwardInference()，swdnnRNNForwardTraining()，swdnnRNNBackwardData()，swdnnRNNBackwardWeights()函数中用于指明循环方向。

1. SWDNN_UNIDIRECTIONAL

2. SWDNN_BIDIRECTIONAL

## swdnnDropoutDescriptor_t

swdnnDropoutDescriptor_t是结构体指针，用于描述dropout操作。通过swdnnCreateDropoutDescriptor()创建一个描述符，通过swdnnSetDropoutDescriptor()初始化这个描述符。通过swdnnGetDropoutDescriptor()查询之前初始化的描述符,通过swdnnRestoreDropoutDescriptor()重新将描述符存储到之前的状态。通过swdnnDestroyDropoutDescriptor()来销毁这个描述符。

## swdnnEmbeddingDescriptor_t

swdnnEmbeddingDescriptor_t是结构体指针，用于描述Embedding操作。通过
swdnnCreateEmbeddingDescriptor()创建一个描述符，通过swdnnSetEmbeddingDescriptor()初始化这个描述符。通过swdnnDestroyEmbeddingDescriptor()来销毁这个描述符。

## swdnnEmbeddingArrayType_t

swdnnEmbeddingArrayType_t是矩阵数据类型。

1. SWDNN_EMBEDDING_ARRAY_DENSE 稠密矩阵

2. SWDNN_EMBEDDING_ARRAY_SPARSE 稀疏矩阵，暂时不支持

## swdnnEmbeddingScaleGradMode_t

swdnnEmbeddingScaleGradMode_t是数据类型。

1. SWDNN_EMBEDDING_SCALE_GRAD_FRED

2. SWDNN_EMBEDDING_SCALE_GRAD_UNFRED


## swdnnFilterDescriptor_t

swdnnFilterDescriptor_t是结构体指针，用于描述卷积核数据集。通过.
swdnnCreateFilterDescriptor()创建描述符，通过swdnnSetFilter4dDescriptor()或swdnnSetFilterNdDescriptor()
初始化被创建的描述符。通过swdnnGetFilter4dDescriptor()或swdnnGetFilterNdDescriptor()获取描述符，通过swdnnDestoryFilterDescriptor()销毁描述符。

## swdnnFoldingDirection_t

此枚举类型定义了fold的2个方向：fold、unfold。

1. SWDNN_TRANSFORM_FOLD

2. SWDNN_TRANSFORM_UNFOLD

## swdnnHandle_t

swdnnHandle_t是个结构体指针，用于描述swdnn库的运行环境和资源情况。在应用开始时通过swdnnCreate()创建并返回这个指针，在结束时通过swdnnDestroy()来释放运行资源。

## swdnnLayerMode_t

swdnnLayerMode_t是枚举变量，用于表明Layernorm所支持的模式。

1. SWDNN_LAYER_NORM_0

2. SWDNN_LAYER_NORM_1（暂不支持）

## swdnnMathType_t

swdnnMathType_t是枚举变量，用于表明库函数是否允许混合精度计算加速。

1. SWDNN_DEFAULT_MATH 默认精度计算

2. SWDNN_TENSOR_ACC_MATH 混合精度计算

3. SWDNN_TENSOR_ACC_MATH_ALLOW_CONVERSION 转换精度计算

## swdnnMultiHeadAttnWeightKind_t

swdnnMultiHeadAttnWeightKind_t是枚举型变量，用于描述多头网络中权重类型。

1. SWDNN_MH_ATTN_Q\_WEIGHTS

2. SWDNN_MH_ATTN_K\_WEIGHTS

3. SWDNN_MH_ATTN_V\_WEIGHTS

4. SWDNN_MH_ATTN_O\_WEIGHTS

5. SWDNN_MH_ATTN_Q\_BIASES

6. SWDNN_MH_ATTN_K\_BIASES

7. SWDNN_MH_ATTN_V\_BIASES

8. SWDNN_MH_ATTN_O\_BIASES

## swdnnNanPropagation_t

swdnnNanPropagation_t是枚举变量，用于表明库函数是否允许传播非数。

1. SWDNN_NOT_PROPAGATE_NAN 不传播非数

2. SWDNN_PROPAGATE_NAN 传播非数

## swdnnPoolingDescriptor_t

swdnnPoolingDescriptor_t是结构体指针，用于描述池化操作。通过swdnnCreatePoolingDescriptor()创建一个描述符，并通过swdnnSetPoolingNdDescriptor()或者swdnnSetPooling2dDescriptor()初始化这个被创建的描述符。通过swdnnGetPooling2dDescriptor()或swdnnGetPoolingNdDescriptor()获取初始化的描述符。通过swdnnDestoryPoolingDescriptor()销毁这个描述符。

## swdnnPoolingMode_t

swdnnPoolingMode_t是枚举变量，用于选择池化操作的类型。

1. SWDNN_POOLING_MAX

2. SWDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING

3. SWDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING

4. SWDNN_POOLING_MAX_DETERMINISTIC

## swdnnRNNAlgo_t

swdnnRNNAlgo_t是枚举型变量，用于选择rnn算法类型。

1. SWDNN_RNN_ALGO_STANDARD（只支持这一种）

2. SWDNN_RNN_ALGO_PERSIST_STATIC

3. SWDNN_RNN_ALGO_PERSIST_DYNAMIC

4. SWDNN_RNN_ALGO_COUNT

## swdnnRNNBiasMode_t

swdnnRNNBiasMode_t是枚举变量，用于配置RNN函数中的偏置向量。

1. SWDNN_RNN_NO_BIAS

2. SWDNN_RNN_SINGLE_INP_BIAS

3. SWDNN_RNN_DOUBLE_BIAS（只支持这一种）

4. SWDNN_RNN_SINGLE_REC_BIAS

## swdnnRNNDescriptor_t

swdnnRNNDescriptor_t是结构体指针，用于描述RNN操作。通过swdnnCreateRNNDescriptor()创建一个描述符，并通过swdnnSetRNNDescriptor()初始化这个描述符。通过swdnnGetRNNDescriptor()来查询这个初始化的描述符,并通过swdnnDestoryRNNDescriptor()来销毁这个描述符。

## swdnnRNNInputMode_t

swdnnRNNInputMode_t是枚举变量，在swdnnRNNForwardInference()，swdnnRNNForwardTraining()，swdnnRNNBackwardData()，swdnnRNNBackwardWeights()函数中用于指明第一层的操作。

1. SWDNN_LINEAR_INPUT

2. SWDNN_SKIP_INPUT

## swdnnRNNMode_t

swdnnRNNMode_t是枚举变量，在swdnnRNNForwardInference()，swdnnRNNForwardTraining()，swdnnRNNBackwardData()，swdnnRNNBackwardWeights()函数中用于指明网络类型。

1. SWDNN_RNN_RELU

2. SWDNN_RNN_TANH

3. SWDNN_LSTM

4. SWDNN_GRU

## swdnnSeqDataDescriptor_t

swdnnSeqDataDescriptor_t 是结构体指针，用于描述序列梯度操作。通过
swdnnCreateSeqDataDescriptor()创建一个描述符，通过swdnnSetSeqDataDescriptor()初始化这个描述符，也可以通过swdnnGetSeqDataDescriptor()查询该描述符。通过swdnnDestroySeqDataDescriptor()销毁该描述符。

## swdnnSeqDataAxis_t

swdnnSeqDataAxis_t是枚举类型变量，用于配置swdnnSetSeqDataDescriptor()中的参数dimA。

1. SWDNN_SEQDATA_TIME_DIM

2. SWDNN_SEQDATA_BATCH_DIM

3. SWDNN_SEQDATA_BEAM_DIM

4. SWDNN_SEQDATA_VECT_DIM

## swdnnSoftmaxAlgorithm_t

swdnnSoftmaxAlgorithm_t是枚举变量，用于选择softmax算法的类型。

1. SWDNN_SOFTMAX_FAST（暂不支持）

2. SWDNN_SOFTMAX_ACCURATE

3. SWDNN_SOFTMAX_LOG（v1.4.1支持）

## swdnnSoftmaxMode_t

swdnnSoftmaxMode_t是枚举变量，用于选择softmax操作的类型。

1. SWDNN_SOFTMAX_MODE_INSTANCE

2. SWDNN_SOFTMAX_MODE_CHANNEL

## swdnnStatus_t

swdnnStatus_t是枚举变量，用于描述库函数的**返回值**。

1. SWDNN_STATUS_SUCCESS

2. SWDNN_STATUS_NOT_INITIALIZED

3. SWDNN_STATUS_ALLOC_FAILED

4. SWDNN_STATUS_BAD_PARAM

5. SWDNN_STATUS_INTERNAL_ERROR

6. SWDNN_STATUS_INVALID_VALUE

7. SWDNN_STATUS_ARCH_MISMATCH

8. SWDNN_STATUS_MAPPING_ERROR

9. SWDNN_STATUS_EXECUTION_FAILED

10. SWDNN_STATUS_NOT_SUPPORTED

11. SWDNN_STATUS_LICENSE_ERROR

12. SWDNN_STATUS_RUNTIME_PREREQUISITE_MISSING

13. SWDNN_STATUS_RUNTIME_IN_PROGRESS

14. SWDNN_STATUS_RUNTIME_FP_OVERFLOW

## swdnnTensorDescriptor_t

swdnnTensorDescriptor_t是结构体指针，指向n维数据集。通过swdnnCreateTensorDescriptor()创建一个描述符，通过swdnnSetTensorNdDescriptor()或者swdnnSetTensor4dDescriptor或者swdnnSetTensor4dDescriptorEx()进行初始化。

## swdnnTensorFormat_t

swdnnTensorFormat_t是枚举变量，用于选择张量存储格式的类型。

1. SWDNN_TENSOR_NCHW

2. SWDNN_TENSOR_NHWC

3. SWDNN_TENSOR_CHWN

## swdnnTensorTransformDescriptor_t

swdnnTensorTransformDescriptor_t是结构体指针，通过swdnnCreateTensorTransformDescriptor()来创建一个描述符，并通过swdnnSetTensorTransformDescriptor()来初始化这个被创建的描述符。通过swdnnGetTensorTransformDescriptor()获取这个被初始化的描述符，通过swdnnDestoryTensorTransformDescriptor()销毁这个描述符。

## swdnnWgradMode_t

swdnnWgradMode_t是枚举类型变量，用于选择梯度更新模式。

1. SWDNN_WGRAD_MODE_ADD

2. SWDNN_WGRAD_MODE_SET

## swdnnReduceTensorDescriptor_t

swdnnReduceTensorDescriptor_t是结构体指针，通过swdnnCreateReduceTensorDescriptor()来创建一个描述符，并通过swdnnSetReduceTensorDescriptor()来初始化这个被创建的描述符。通过swdnnGetReduceTensorDescriptor()获取这个被初始化的描述符，通过swdnnDestoryReduceTensorDescriptor()销毁这个描述符。

## swdnnReduceTensorOp_t

swdnnReduceTensorOp_t是枚举类型变量，用于选择reduce模式。

1. SWDNN_REDUCE_TENSOR_ADD

2. SWDNN_REDUCE_TENSOR_MUL（MUL的规约操作，要注意规约顺序不同导致的精度问题）

3. SWDNN_REDUCE_TENSOR_MIN

4. SWDNN_REDUCE_TENSOR_MAX

5. SWDNN_REDUCE_TENSOR_AMAX

6. SWDNN_REDUCE_TENSOR_AVG

7. SWDNN_REDUCE_TENSOR_NORM1

8. SWDNN_REDUCE_TENSOR_NORM2

9. SWDNN_REDUCE_TENSOR_MUL_NO_ZEROS

## swdnnReduceTensorIndices_t

swdnnReduceTensorIndices_t是枚举类型变量，用于reduce的MAX，AMAX，MIN模式中选择索引模式。

1. SWDNN_REDUCE_TENSOR_NO_INDICES

2. SWDNN_REDUCE_TENSOR_FLATTENED_INDICES

## swdnnOpTensorDescriptor_t

swdnnOpTensorDescriptor_t是结构体指针，通过swdnnCreateOpTensorDescriptor()来创建一个描述符，并通过swdnnSetOpTensorDescriptor()来初始化这个被创建的描述符。通过swdnnGetOpTensorDescriptor()获取这个被初始化的描述符，通过swdnnDestoryOpTensorDescriptor()销毁这个描述符。

## **swdnnIndicesType_t**

swdnnIndicesType_t是枚举类型变量，用于reduce的索引返回值的类型。

1. SWDNN_32BIT_INDICES

2. SWDNN_64BIT_INDICES

3. SWDNN_16BIT_INDICES

4. SWDNN_8BIT_INDICES

## **swdnnOpTensorOp_t**

swdnnOpTensorOp_t是枚举类型变量，用于选择计算模式。

1. SWDNN_OP_TENSOR_ADD

2. SWDNN_OP_TENSOR_MUL

3. SWDNN_OP_TENSOR_MIN

4. SWDNN_OP_TENSOR_MAX

5. SWDNN_OP_TENSOR_SQRT

6. SWDNN_OP_TENSOR_NOT

## swdnnFusedOpsPlan_t

swdnnFusedOpsPlan_t是结构体指针，通过swdnnCreateFusedOpsPlan()来创建一个描述符，并通过swdnnMakeFusedOpsPlan()来初始化这个被创建的描述符，通过swdnnDestroyFusedOpsPlan()销毁这个描述符。

## **swdnnFusedOp_t**

swdnnFusedOps_t是枚举类型变量，用于选择融合计算模式。

1. SWDNN_FUSED_CONV_BN_STATISTICS

2. SWDNN_FUSED_PERMUTE_BN_FINALIZE_ACTIVATION

3. SWDNN_FUSED_ACTIVATION_BN_BACKWARD

4. SWDNN_FUSED_BN_ACTIVATION_FORWARD

5. SWDNN_FUSED_ADD_ACTIVATION_FORWARD

5. SWDNN_FUSED_CONV_BIAS_ACTIVATION_FORWARD

## **swdnnFusedOpsConstParamLabel_t**

swdnnFusedOpsConstParamLabel_t是枚举类型变量，用于选择传入的描述符类型。

1. SWDNN_PARAM_XDESC
2. SWDNN_PARAM_DXDESC
3. SWDNN_PARAM_YDESC
4. SWDNN_PARAM_DYDESC
5. SWDNN_PARAM_WDESC
6. SWDNN_PARAM_DWDESC
7. SWDNN_PARAM_CONV_DESC
8. SWDNN_PARAM_BN_MODE
9. SWDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC
10. SWDNN_PARAM_ACTIVATION_DESC
10. SWDNN_PARAM_BIASDESC

## **swdnnFusedOpsVariantParamLabel_t**

swdnnFusedOpsVariantParamLabel_t\_t是枚举类型变量，用于选择传入的参数类型。

1. SWDNN_PTR_XDATA
2. SWDNN_PTR_DXDATA
3. SWDNN_PTR_WDATA
4. SWDNN_PTR_DWDATA
5. SWDNN_PTR_CONV_YDATA
6. SWDNN_PTR_CONV_DYDATA
7. SWDNN_PTR_CONV_YDATA_5D
8. SWDNN_PTR_CONV_DYDATA_5D
9. SWDNN_SCALE_DOUBLE_BN_EPSILON
10. SWDNN_SCALE_DOUBLE_BN_EXP_AVG_FACTOR
11. SWDNN_PTR_BN_RUNNING_MEAN
12. SWDNN_PTR_BN_RUNNING_VAR
13. SWDNN_PTR_BN_SAVED_MEAN
14. SWDNN_PTR_BN_SAVED_INVSTD
15. SWDNN_PTR_BN_SCALE
16. SWDNN_PTR_BN_DSCALE
17. SWDNN_PTR_BN_BIAS
18. SWDNN_PTR_BN_DBIAS
19. SWDNN_PTR_BN_YDATA
20. SWDNN_PTR_BN_DYDATA
21. SWDNN_PTR_ACT_YDATA
22. SWDNN_PTR_ACT_DYDATA
23. SWDNN_PTR_ADD_ALPHA
24. SWDNN_PTR_ADD_BETA
25. SWDNN_PTR_YDATA
26. SWDNN_PTR_DYDATA
27. SWDNN_PTR_CONV_BIAS
28. SWDNN_SCALE_SIZE_T_WORKSPACE_SIZE_IN_BYTES
29. SWDNN_PTR_WORKSPACE
29. SWDNN_PTR_CONV_BIAS

## swdnnUnaryOpsMode_t

swdnnUnaryOpsMode_t是枚举类型变量，用于选择swdnnUnaryOps的计算模式

```c
    // without alpha( y = ops(x) )
    SWDNN_BATCH_LOG = 0,
    SWDNN_BATCH_EXP = 1,
    SWDNN_BATCH_SQRT = 2,
    SWDNN_BATCH_RSQRT = 3,
    SWDNN_BATCH_SQUARE = 4,
    SWDNN_BATCH_SIN = 5,
    SWDNN_BATCH_COS = 6,
    SWDNN_BATCH_TANH = 7,
    SWDNN_BATCH_CEIL = 8,
    SWDNN_BATCH_FLOOR = 9,
    SWDNN_BATCH_FABS = 10,

    // with alpha( y = ops(alpha*x) )
    SWDNN_BATCH_ADD_A = 11,
    SWDNN_BATCH_SUB_A = 12,
    SWDNN_BATCH_MUL_A = 13,
    SWDNN_BATCH_DIV_A = 14,
    SWDNN_BATCH_RDIV = 15,
    SWDNN_BATCH_POW = 16,

    // convert 
    SWDNN_BATCH_S2H = 17,
    SWDNN_BATCH_H2S = 18
```

## swdnnInplaceOpsMode_t

swdnnInplaceOpsMode_t是枚举类型变量，用于选择swdnnInplaceOps的计算模式

```c
    SWDNN_BATCH_LOG = 0, //y=log(y)
    SWDNN_BATCH_EXP = 1, //y=exp(y)
    SWDNN_BATCH_SQRT = 2,  //y=sqrt(y)
    SWDNN_BATCH_SQUARE = 4, //y=square(y)
    SWDNN_BATCH_SIN = 5, //y=square(y)
    SWDNN_BATCH_COS = 6, //y=cos(y)
    SWDNN_BATCH_TANH = 7, //y=tanh(y)
    SWDNN_BATCH_CEIL = 8, //y=ceil(y)
    SWDNN_BATCH_FLOOR = 9, //y=floor(y)
    SWDNN_BATCH_FABS = 10, //y=fabs(y)
    SWDNN_BATCH_POW = 16, //y=pwd(y,x)
    SWDNN_BATCH_SIGMOID = 19, //y=sigmoid(y)
    SWDNN_BATCH_RELU = 20, //y=relu(y)
    SWDNN_BATCH_ELU = 21, //y=elu(y, alpha)
    SWDNN_BATCH_GELU = 22
```

## swdnnAdaptivePoolingMode_t

swdnnAdaptivePoolingMode_t是枚举类型变量，用于选择自适应池化的计算模式

```c
    SWDNN_ADAPTIVE_POOLING_MAX = 0,
    SWDNN_ADAPTIVE_POOLING_AVG = 1
```

## swdnnLossReductionMode_t

swdnnLossReductionMode_t是枚举类型变量，用于选择Loss的reduce模式

```c
    SWDNN_LOSS_REDUCTION_NONE = 0,
    SWDNN_LOSS_REDUCTION_MEAN = 1,
    SWDNN_LOSS_REDUCTION_SUM = 2,
    SWDNN_LOSS_REDUCTION_BATCH_MEAN = 3
```

## swdnnCornerPoolMode_t

swdnnCornerPoolMode_t是枚举类型变量，用于选择corner池化的模式

```c
    SWDNN_CORNER_POOL_BOTTOM = 0,
    SWDNN_CORNER_POOL_TOP = 1,
    SWDNN_CORNER_POOL_RIGHT = 2,
    SWDNN_CORNER_POOL_LEFT = 3,
```

# API列表

## swdnnActivationBackward

```C
swdnnStatus_t swdnnActivationBackward(
	swdnnHandle_t handle,
	swdnnActivationDescriptor_t activationDesc,
	const void *alpha,
	const swdnnTensorDescriptor_t yDesc,
	const void *y,
	const swdnnTensorDescriptor_t dyDesc,
	const void *dy,
	const swdnnTensorDescriptor_t xDesc,
	const void *x,
	const void *beta,
	const swdnnTensorDescriptor_t dxDesc,
	void *dx)
```

**功能描述**：

反向过程中激活函数的梯度值计算

**参数描述：**

-	handle：输入，控制句柄 
-	activationDesc：输入，激活描述符 
-	alpha, beta：输入，扩展因子 
-	yDesc：输入，y的张量描述符 
-	y：输入，y的首地址指针 
-	dyDesc：输入，dy的张量描述符 
-	dy：输入，y的首地址指针 
-	xDesc：输入，x的张量描述符 
-	x：输入，x的首地址指针 
-	dxDesc：输入，dx的张量描述符 
-	dx：输出，dx的首地址指针

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC

2.  SWDNN_TENSOR_NCHW

3.  SWDNN_TENSOR_CHWN

**目前已支持的激活模式swdnnActivationMode_t：**

1. SWDNN_ACTIVATION_SIGMOID

2. SWDNN_ACTIVATION_RELU

3. SWDNN_ACTIVATION_TANH

4. SWDNN_ACTIVATION_CLIPPED_RELU

5. SWDNN_ACTIVATION_ELU

6. SWDNN_ACTIVATION_SIGMOID_TAB

7. SWDNN_ACTIVATION_ELU_TAB

8. SWDNN_ACTIVATION_TANH_TAB

## swdnnActivationForward

```C
swdnnStatus_t swdnnActivationForward( 
	swdnnHandle_t handle, 
	swdnnActivationDescriptor_t activationDesc, 
	const void *alpha, 
	const swdnnTensorDescriptor_t xDesc, 
	const void *x, 
	const void *beta, 
	const swdnnTensorDescriptor_t yDesc, 
	void *y)
```

**功能描述**：

正向过程中激活函数的激活值计算

**参数描述**：

-	handle：输入，控制句柄 
-	activationDesc：输入，激活描述符 
-	alpha, beta：输入，扩展因子 
-	xDesc：输入，x的张量描述符 
-	x：输入，x的首地址指针 
-	yDesc：输入，y的张量描述符 
-	y：输出，y的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC

2.  SWDNN_TENSOR_NCHW

3.  SWDNN_TENSOR_CHWN

**目前已支持的激活模式swdnnActivationMode_t：**

1. SWDNN_ACTIVATION_SIGMOID

2. SWDNN_ACTIVATION_RELU

3. SWDNN_ACTIVATION_TANH

4. SWDNN_ACTIVATION_CLIPPED_RELU

5. SWDNN_ACTIVATION_ELU

6. SWDNN_ACTIVATION_SIGMOID_TAB

7. SWDNN_ACTIVATION_ELU_TAB

8. SWDNN_ACTIVATION_TAHN_TAB

备注：6、7、8、所支持的三种TAB模式仅支持float类型数据

## swdnnAddTensor

```C
swdnnStatus_t swdnnAddTensor(
	swdnnHandle_t handle,
	const void *alpha,
	const swdnnTensorDescriptor_t aDesc,
	const void *A,
	const void *beta,
	const swdnnTensorDescriptor_t cDesc,
	void *C)
```

**功能描述**：

张量相加计算

**参数描述**：

-	handle：输入，控制句柄 
-	alpha, beta：输入，扩展因子 
-	aDesc：输入，A的张量描述符 
-	A：输入，A的首地址指针 
-	cDesc：输入，C的张量描述符 
-	C：输入/输出，C的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC

2.  SWDNN_TENSOR_NCHW

3.  SWDNN_TENSOR_CHWN

## swdnnAddTensorEx

```C
swdnnAddTensorEx(
    swdnnHandle_t handle, 
    const swdnnTensorDescriptor_t aDesc,
    const void *A, 
    const swdnnTensorDescriptor_t bDesc, 
    const void *B,
    const swdnnTensorDescriptor_t cDesc, void *C)
```

**功能描述**：

张量相加计算 C\[i\]=A\[i\]+B\[i\]，支持双向广播，即允许张量A和B的某维度为1，张量C的维度取A和B的最大维度。
比如：A(1, 16, 1, 8)，B(2, 16, 8, 8)，则C的维度为(2, 16, 8, 8)

**参数描述**：

-	handle：输入，控制句柄 
-	aDesc：输入，A的张量描述符 
-	A：输入，A的首地址指针 
-	bDesc：输入，A的张量描述符 
-	B：输入，A的首地址指针 
-	cDesc：输入，C的张量描述符 
-	C：输出，C的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC
2.  SWDNN_TENSOR_NCHW

**其他限制**
对于SWDNN_DATA_HALF类型，要求三个张量的H、W、C三个维度的乘积为偶数，即H\*W\*C % 2 == 0

## swdnnDivTensorEx

```C
swdnnDivTensorEx(
    swdnnHandle_t handle, 
    const swdnnTensorDescriptor_t aDesc,
    const void *A, 
    const swdnnTensorDescriptor_t bDesc, 
    const void *B,
    const swdnnTensorDescriptor_t cDesc, void *C)
```

**功能描述**：

张量除法计算 C\[i\]=A\[i\]/B\[i\]，支持双向广播，即允许张量A和B的某维度为1，张量C的维度取A和B的最大维度。
比如：A(1, 16, 1, 8)，B(2, 16, 8, 8)，则C的维度为(2, 16, 8, 8)

**参数描述**：

-	handle：输入，控制句柄 
-	aDesc：输入，A的张量描述符 
-	A：输入，A的首地址指针 
-	bDesc：输入，A的张量描述符 
-	B：输入，A的首地址指针 
-	cDesc：输入，C的张量描述符 
-	C：输出，C的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC
2.  SWDNN_TENSOR_NCHW

**其他限制**
对于SWDNN_DATA_HALF类型，要求三个张量的H、W、C三个维度的乘积为偶数，即H\*W\*C % 2 == 0

## swdnnMulTensorEx

```C
swdnnMulTensorEx(
    swdnnHandle_t handle, 
    const swdnnTensorDescriptor_t aDesc,
    const void *A, 
    const swdnnTensorDescriptor_t bDesc, 
    const void *B,
    const swdnnTensorDescriptor_t cDesc, void *C)
```

**功能描述**：

张量乘法计算 C\[i\]=A\[i\]*B\[i\]，支持双向广播，即允许张量A和B的某维度为1，张量C的维度取A和B的最大维度。
比如：A(1, 16, 1, 8)，B(2, 16, 8, 8)，则C的维度为(2, 16, 8, 8)

**参数描述**：

-	handle：输入，控制句柄 
-	aDesc：输入，A的张量描述符 
-	A：输入，A的首地址指针 
-	bDesc：输入，A的张量描述符 
-	B：输入，A的首地址指针 
-	cDesc：输入，C的张量描述符 
-	C：输出，C的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC
2.  SWDNN_TENSOR_NCHW

**其他限制**
对于SWDNN_DATA_HALF类型，要求三个张量的H、W、C三个维度的乘积为偶数，即H\*W\*C % 2 == 0

## swdnnSubTensorEx

```C
swdnnSubTensorEx(
    swdnnHandle_t handle, 
    const swdnnTensorDescriptor_t aDesc,
    const void *A, 
    const swdnnTensorDescriptor_t bDesc, 
    const void *B,
    const swdnnTensorDescriptor_t cDesc, void *C)
```

**功能描述**：

张量减法计算 C\[i\]=A\[i\]-B\[i\]，支持双向广播，即允许张量A和B的某维度为1，张量C的维度取A和B的最大维度。
比如：A(1, 16, 1, 8)，B(2, 16, 8, 8)，则C的维度为(2, 16, 8, 8)

**参数描述**：

-	handle：输入，控制句柄 
-	aDesc：输入，A的张量描述符 
-	A：输入，A的首地址指针 
-	bDesc：输入，A的张量描述符 
-	B：输入，A的首地址指针 
-	cDesc：输入，C的张量描述符 
-	C：输出，C的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC
2.  SWDNN_TENSOR_NCHW

**其他限制**
对于SWDNN_DATA_HALF类型，要求三个张量的H、W、C三个维度的乘积为偶数，即H\*W\*C % 2 == 0

## swdnnBatchNormalizationBackward

```C
swdnnStatus_t swdnnBatchNormalizationBackward(
	swdnnHandle_t handle,
	swdnnBatchNormMode_t mode,
	const void *alphaDataDiff,
	const void *betaDataDiff,
	const void *alphaParamDiff,
	const void *betaParamDiff,
	const swdnnTensorDescriptor_t xDesc,
	const void *x,
	const swdnnTensorDescriptor_t dyDesc,
	const void *dy,
	const swdnnTensorDescriptor_t dxDesc,
	void *dx,
	const swdnnTensorDescriptor_t bnScaleBiasDiffDesc,
	const void *bnScale,
	void *resultBnScaleDiff,
	void *resultBnBiasDiff,
	double epsilon,
	const void *savedMean,
	const void *savedInvVariance)
```

**功能描述**：

反向过程中的BN计算

**参数描述**：

-	handle：输入，控制句柄 
-	mode：输入，BN的模式 
-	alphaDataDiff, betaDataDiff：输入，x的扩展因子 
-	alphaParamDiff, betaParamDiff：输入，resultBnScaleDiff和resultBnBiasDiff的扩展因子 
-	xDesc：输入，x的张量描述符 
-	x：输入，x的首地址指针 
-	yDesc：输入，y的张量描述符 
-	y：输入，y的首地址指针 
-	dyDesc：输入，dy的张量描述符 
-	dy：输入，dy的首地址指针 
-	dxDesc：输出，dx的张量描述符 
-	dx：输出，dx的首地址指针 
-	bnScaleBiasDiffDesc：输入，scale, bias, mean, var的张量描述符 
-	bnScale：输入，bnScale的首地址指针 
-	resultBnScaleDiff：输出，resultBnScaleDiff的首地址指针 
-	resultBnBiasDiff：输出，resultBnBiasDiff的首地址指针 
-	epsilon：输入，BN公式中的epsilon值 
-	savedMean：输入，savedMean的首地址指针 
-	savedInvVariance：输入，savedInvVariance的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1. SWDNN_TENSOR_NHWC

**目前已支持的BN层正向推理计算swdnnBatchNormMode_t mode：**

1. SWDNN_BATCHNORM_PER_ACTIVATION

2. SWDNN_BATCHNORM_SPATIAL

**其他限制：**

1. epsilon\>0

## swdnnBatchNormalizationForwardInference

```C
swdnnStatus_t swdnnBatchNormalizationForwardInference(
	swdnnHandle_t handle,
	swdnnBatchNormMode_t mode,
	const void *alpha,
	const void *beta,
	const swdnnTensorDescriptor_t xDesc,
	const void *x,
	const swdnnTensorDescriptor_t yDesc,
	void *y,
	const swdnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
	const void *bnScale,
	const void *bnBias,
	const void *estimatedMean,
	const void *estimatedVariance,
	double epsilon)
```

**功能描述**：

推理过程中的BN计算

**参数描述**：

-	 handle：输入，控制句柄 
-	 mode：输入，BN的模式 
-	 alpha, beta：输入，扩展因子，目前未用 
-	 xDesc：输入，x的张量描述符 
-	 x：输入，x的首地址指针 
-	 yDesc：输入，y的张量描述符 
-	 y：输出，y的首地址指针 
-	 bnScaleBiasMeanVarDesc：输入，bnScale, bnBias, estimatedMean, estimatedVariance的张量描述符
-	 bnScale：输入，bnScale的首地址指针
-	 bnBias：输入，bnBias的首地址指针
-	 estimatedMean：输入，estimatedMean的首地址指针
-	 estimatedVariance：输入，estimatedVariance的首地址指针
-	 epsilon：输入，BN公式中的epsilon值

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1. SWDNN_TENSOR_NHWC

**目前已支持的BN层正向推理计算swdnnBatchNormMode_t mode：**

1. SWDNN_BATCHNORM_PER_ACTIVATION

2. SWDNN_BATCHNORM_SPATIAL

**其他限制：**

1. epsilon\>0

## swdnnBatchNormalizationForwardTraining

```C
swdnnStatus_t swdnnBatchNormalizationForwardTraining(
	swdnnHandle_t handle,
	swdnnBatchNormMode_t mode,
	const void *alpha,
	const void *beta,
	const swdnnTensorDescriptor_t xDesc,
	const void *x,
	const swdnnTensorDescriptor_t yDesc,
	void *y,
	const swdnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
	const void *bnScale,
	const void *bnBias,
	float exponentialAverageFactor,
	void *resultRunningMean,
	void *resultRunningVariance,
	double epsilon,
	void *resultSaveMean,
	void *resultSaveInvVariance)
```

**功能描述**：

训练过程中的BN计算

**参数描述**：

-	handle：输入，控制句柄 
-	mode：输入，BN的模式 
-	alpha, beta：输入，扩展因子，目前未用 
-	xDesc：输入，x的张量描述符 
-	x：输入，x的首地址指针 
-	yDesc：输入，y的张量描述符 
-	y：输出，y的首地址指针 
-	bnScaleBiasMeanVarDesc：输入，bnScale, bnBias, estimatedMean, estimatedVariance的张量描述符
-	bnScale：输入，bnScale的首地址指针
-	bnBias：输入，bnBias的首地址指针
-	exponentialAverageFactor：输入，exponentialAverageFactor的首地址指针 
-	resultRunningMean：输入，resultRunningMean的首地址指针 
-	resultRunningVariance：输入，resultRunningVariance的首地址指针 
-	epsilon：输入，BN公式中的epsilon值 
-	resultSaveMean：输入，resultSaveMean的首地址指针 
-   resultSaveInvVariance：输入，resultSaveInvVariance的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1. SWDNN_TENSOR_NHWC

**目前已支持的BN层正向推理计算swdnnBatchNormMode_t：**

1. SWDNN_BATCHNORM_PER_ACTIVATION

2. SWDNN_BATCHNORM_SPATIAL

**其他限制：**

1. epsilon\>0；

## swdnnBitwiseAndTensor

```C
swdnnBitwiseAndTensor(
    swdnnHandle_t handle, 
    const swdnnTensorDescriptor_t aDesc,
    const void *A, 
    const swdnnTensorDescriptor_t bDesc, 
    const void *B, 
    const swdnnTensorDescriptor_t cDesc,
    void *C);

```

**功能描述**：

张量逐比特与运算，C\[i\]=A\[i\]&B\[i\], 支持双向广播，即允许张量A和B的某维度为1，张量C的维度取A和B的最大维度。
比如：A(1, 16, 1, 8)，B(2, 16, 8, 8)，则C的维度为(2, 16, 8, 8)

**参数描述**：

-	handle：输入，控制句柄 
-	aDesc：输入，A的张量描述符 
-	A：输入，A的首地址指针 
-	bDesc：输入，A的张量描述符 
-	B：输入，A的首地址指针 
-	cDesc：输入，C的张量描述符 
-	C：输出，C的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_INT32

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC
2.  SWDNN_TESNOR_NCHW
3.  SWDNN_TESNOR_CNHW

## swdnnBitwiseNotTensor

```C
swdnnBitwiseNotTensor(
    swdnnHandle_t handle, 
    const swdnnTensorDescriptor_t aDesc,
    const void *A, 
    const swdnnTensorDescriptor_t cDesc,
    void *C);
```

**功能描述**：

张量逐比特取反运算，C\[i\]=!A\[i\]。

**参数描述**：

-	handle：输入，控制句柄 
-	aDesc：输入，A的张量描述符 
-	A：输入，A的首地址指针 
-	cDesc：输入，C的张量描述符 
-	C：输出，C的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_INT32

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC
2.  SWDNN_TESNOR_NCHW
3.  SWDNN_TESNOR_CNHW

## swdnnBitwiseOrTensor

```C
swdnnBitwiseOrTensor(
    swdnnHandle_t handle, 
    const swdnnTensorDescriptor_t aDesc,
    const void *A, 
    const swdnnTensorDescriptor_t bDesc, 
    const void *B, 
    const swdnnTensorDescriptor_t cDesc,
    void *C);

```

**功能描述**：

张量逐比特或运算，C\[i\]=A\[i\] | B\[i\], 支持双向广播，即允许张量A和B的某维度为1，张量C的维度取A和B的最大维度。
比如：A(1, 16, 1, 8)，B(2, 16, 8, 8)，则C的维度为(2, 16, 8, 8)

**参数描述**：

-	handle：输入，控制句柄 
-	aDesc：输入，A的张量描述符 
-	A：输入，A的首地址指针 
-	bDesc：输入，A的张量描述符 
-	B：输入，A的首地址指针 
-	cDesc：输入，C的张量描述符 
-	C：输出，C的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_INT32

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC
2.  SWDNN_TESNOR_NCHW
3.  SWDNN_TESNOR_CNHW

## swdnnBitwiseXorTensor

```C
swdnnBitwiseXorTensor(
    swdnnHandle_t handle, 
    const swdnnTensorDescriptor_t aDesc,
    const void *A, 
    const swdnnTensorDescriptor_t bDesc, 
    const void *B, 
    const swdnnTensorDescriptor_t cDesc,
    void *C);

```

**功能描述**：

张量逐比特易或运算，C\[i\]=A\[i\] ^ B\[i\], 支持双向广播，即允许张量A和B的某维度为1，张量C的维度取A和B的最大维度。
比如：A(1, 16, 1, 8)，B(2, 16, 8, 8)，则C的维度为(2, 16, 8, 8)

**参数描述**：

-	handle：输入，控制句柄 
-	aDesc：输入，A的张量描述符 
-	A：输入，A的首地址指针 
-	bDesc：输入，A的张量描述符 
-	B：输入，A的首地址指针 
-	cDesc：输入，C的张量描述符 
-	C：输出，C的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_INT32

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC
2.  SWDNN_TESNOR_NCHW
3.  SWDNN_TESNOR_CNHW

## swdnnClipTensor

```C
swdnnClipTensor(swdnnHandle_t handle, 
	float min_value, 
	float max_value, 
	const swdnnTensorDescriptor_t aDesc, 
	void *a, 
	const swdnnTensorDescriptor_t cDesc, 
	void *c)
```

**功能描述：**

进行ClipTensor计算

**参数描述**：

-   handle：输入，设备句柄
-   min_value：输入，截断的下界
-   max_value：输入，截断的上界
-   aDesc：输入，输入数据a的描述符
-   a：输入，数据a
-   cDesc：输出，输出数据c的描述符
-   c：输出，数据c

**返回值**：

1.  SWDNN_STATUS_SUCCESS成功

**备注：**

1. 支持float和half计算

## swdnnConvolutionBackwardBias

```C
swdnnStatus_t swdnnConvolutionBackwardBias(
	swdnnHandle_t handle,
	const void *alpha,
	const swdnnTensorDescriptor_t dyDesc,
	const void *dy,
	const void *beta,
	const swdnnTensorDescriptor_t dbDesc,
	void *db)
```

**功能描述**：

反向过程中的偏置计算

**参数描述**：

-	handle：输入，控制句柄 
-	alpha, beta：输入，扩展因子，目前未用 
-	dyDesc：输入，dyDesc的张量描述符，特征图残差 
-	dy：输入，dy的首地址指针，特征图残差 
-	dbDesc：输入，dbDesc的张量描述符，偏置 
-	db：输出，db的首地址指针，偏置

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1. SWDNN_TENSOR_NHWC

## swdnnConvolutionBackwardData

```C
swdnnStatus_t swdnnConvolutionBackwardData(
	swdnnHandle_t handle,
	const void *alpha,
	const swdnnFilterDescriptor_t wDesc,
	const void *w,
	const swdnnTensorDescriptor_t dyDesc,
	const void *dy,
	const swdnnConvolutionDescriptor_t convDesc,
	swdnnConvolutionBwdDataAlgo_t algo,
	void *workSpace,
	size_t workSpaceSizeInBytes,
	const void *beta,
	const swdnnTensorDescriptor_t dxDesc,
	void *dx)
```

**功能描述**：

反向过程中的数据残差计算

**参数描述**：

-	 handle：输入，控制句柄 
-	 alpha, beta：输入，扩展因子，目前未用
-	 wDesc：输入，wDesc的张量描述符，卷积核
-	 w：输入，w的首地址指针, 卷积核
-	 dyDesc：输入，dy的张量描述符
-	 dy：输入，y的首地址指针
-	 convDesc：输入，卷积意义描述符
-	 algo：输入，卷积算法描述符
-	 workSpace：输入，工作空间的首地址
-	 workSpaceSizeInBytes：输入，工作空间大小
-	 dxDesc：输入，dx的张量描述符
-	 dx：输入/输出，dx的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

(1)支持x,w,y任意数据分布

**目前已支持的反向卷积算法swdnnConvolutionBwdDataAlgo_t：**

1. SWDNN_CONVOLUTION_BWD_DATA_ALGO_0

2. SWDNN_CONVOLUTION_BWD_DATA_ALGO_1（暂不支持）

**目前已支持的方向卷积计算swdnnMathType_t：**

1. SWDNN_TENSOR_ACC_MATH

**groupCounvF模式：**

1. groupCount>1；（swdnnSetConvolutionGroupCount）

2. C%groupCount==0 （其中，wDesc中的channel需要是C/groupCount）

3. M%groupCount==0

v1.4.2版本中存在限制，限制如下：

1. (C/groupCount)%32 == 0
2. (M/groupCount)%32 == 0
3. dilation 只支持为1
4. 不支持alpha，beta

**depthwiseConvF模式：**

1. groupCount>1；（swdnnSetConvolutionGroupCount）

2. C == groupCount（其中，wDesc中的channel==1）

3. M == groupCount

v1.4.2版本中存在限制，限制如下：

1. 不支持大数据量的计算
2. dilation 只支持为1
3. 不支持alpha，beta

**conv3dF模式：**

1. convDesc设置的维度是3；（swdnnSetConvolutionNdDescriptor）
2. padA = [pad_h,pad_w,pad_d]
3. strideA = [stride_h, stride_w, stride_d]
4. dilationA = [dilation_h, dilation_w, dilation_d]

2. xDesc是5维向量，dimA = [n,c,h,w,d]，默认存储顺序为NHWDC
3. wDesc是5维向量，dimA = [m,c,r,s,l]，默认存储顺序为CHWM

3. yDescs是5维向量，dimA = [n,m,e,f,v]，默认存储顺序为NEFLM

v1.4.2版本中存在限制，限制如下：

1. C%32 == 0
2. M%32 == 0
3. 不支持alpha，beta

## swdnnConvolutionBackwardFilter

```C
swdnnStatus_t swdnnConvolutionBackwardFilter(
	swdnnHandle_t handle,
	const void *alpha,
	const swdnnTensorDescriptor_t xDesc,
	const void *x,
	const swdnnTensorDescriptor_t dyDesc,
	const void *dy,
	const swdnnConvolutionDescriptor_t convDesc,
	swdnnConvolutionBwdFilterAlgo_t algo,
	void *workSpace,
	size_t workSpaceSizeInBytes,
	const void *beta,
	const swdnnFilterDescriptor_t dwDesc,
	void *dw)
```

**功能描述**：

反向过程中的卷积核更新值计算

**参数描述**：

-	handle：输入，控制句柄 
-	alpha, beta：输入，扩展因子 
-	xDesc：输入，x的张量描述符，图片数据 
-	x：输入，x的首地址指针, 图片数据 
-	dyDesc：输入，dy的张量描述符 
-	dy：输入，y的首地址指针 
-	convDesc：输入，卷积描述符 
-	algo：输入，卷积算法 
-	workSpace：输入，工作空间的首地址 
-	workSpaceSizeInBytes：输入，工作空间大小 
-	dwDesc：输入，dw的张量描述符 
-	dw：输入/输出，dw的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1. SWDNN_DATA_FLOAT

2. SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

(1)支持x,w,y任意数据分布）

**目前已支持的卷积算法swdnnConvolutionBwdFilterAlgo_t：**

1. SWDNN_CONVOLUTION_BWD_FILTER_ALGO \_0

**目前已支持的卷积模式swdnnConvolutionMode_t：**

1. SWDNN_CROSS_CORRELATION

**目前已支持的卷积计算swdnnMathType_t：**

1. SWDNN_TENSOR_ACC_MATH

**groupCounvF模式：**

1. groupCount>1；（swdnnSetConvolutionGroupCount）

2. C%groupCount==0 （其中，wDesc中的channel需要是C/groupCount）

3. M%groupCount==0

v1.4.2版本中存在限制，限制如下：

1. N%32 == 0
2. (C/groupCount)%32 == 0
3. (M/groupCount)%32 == 0
4. dilation 只支持为1
5. 不支持alpha，beta

**depthwiseConvF模式：**

1. groupCount>1；（swdnnSetConvolutionGroupCount）

2. C == groupCount（其中，wDesc中的channel==1）

3. M == groupCount

v1.4.2版本中存在限制，限制如下：

1. dilation 只支持为1
2. 不支持alpha，beta

**conv3dF模式：**

1. convDesc设置的维度是3；（swdnnSetConvolutionNdDescriptor）
2. padA = [pad_h,pad_w,pad_d]
3. strideA = [stride_h, stride_w, stride_d]
4. dilationA = [dilation_h, dilation_w, dilation_d]

2. xDesc是5维向量，dimA = [n,c,h,w,d]，默认存储顺序为NHWDC
3. wDesc是5维向量，dimA = [m,c,r,s,l]，默认存储顺序为CHWM

3. yDescs是5维向量，dimA = [n,m,e,f,v]，默认存储顺序为NEFLM

v1.4.2版本中存在限制，限制如下：

1. C%32 == 0
2. M%32 == 0
3. N%32 == 0
4. 不支持alpha，beta

## swdnnConvolutionForward

```C
swdnnStatus_t swdnnConvolutionForward( 
	swdnnHandle_t handle, 
	const void *alpha, 
	const swdnnTensorDescriptor_t xDesc, 
	const void *x, 
	const swdnnFilterDescriptor_t wDesc, 
	const void *w, 
	const swdnnConvolutionDescriptor_t convDesc, 
	swdnnConvolutionFwdAlgo_t algo, 
	void *workSpace, 
	size_t workSpaceSizeInBytes, 
	const void *beta, 
	const swdnnTensorDescriptor_t yDesc, 
	void *y)
```

**功能描述**：

正向过程中的卷积计算

**参数描述：**

-	handle：输入，控制句柄 
-	alpha, beta：输入，扩展因子 
-	xDesc：输入，x的张量描述符 
-	x：输入，x的首地址指针 
-	wDesc：输入，w的张量描述符 
-	w：输入，w的首地址指针 
-	convDesc：输入，卷积描述符 
-	algo：输入，卷积算法 
-	workSpace：输入，工作空间的首地址 
-	workSpaceSizeInBytes：输入，工作空间大小 
-	yDesc：输入，y的张量描述符 
-	y：输入/输出，y的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据分布swdnnTensorFormat_t：**

(1)支持x,w,y任意数据分布

**目前已支持的卷积算法swdnnConvolutionFwdAlgo_t：**

1. SWDNN_CONVOLUTION_FWD_ALGO_0

**目前已支持的卷积模式swdnnConvolutionMode_t：**

1. SWDNN_CROSS_CORRELATION

**目前已支持的卷积计算swdnnMathType_t：**

1. SWDNN_TENSOR_ACC_MATH

**groupCounvF模式：**

1. groupCount>1；（swdnnSetConvolutionGroupCount）

2. C%groupCount==0 （其中，wDesc中的channel需要是C/groupCount）

3. M%groupCount==0

v1.4.2版本中存在限制，限制如下：

1. (C/groupCount)%32 == 0
2. (M/groupCount)%32 == 0
3. dilation 只支持为1
4. 不支持大数据量的计算
5. 不支持alpha，beta

**depthwiseConvF模式：**

1. groupCount>1；（swdnnSetConvolutionGroupCount）

2. C == groupCount（其中，wDesc中的channel==1）

3. M == groupCount

v1.4.2版本中存在限制，限制如下：

1. 不支持大数据量的计算
2. dilation 只支持为1
3. 不支持alpha，beta

**conv3dF模式：**

1. convDesc设置的维度是3；（swdnnSetConvolutionNdDescriptor）
2. padA = [pad_h,pad_w,pad_d]
3. strideA = [stride_h, stride_w, stride_d]
4. dilationA = [dilation_h, dilation_w, dilation_d]

2. xDesc是5维向量，dimA = [n,c,h,w,d]，默认存储顺序为NHWDC
3. wDesc是5维向量，dimA = [m,c,r,s,l]，默认存储顺序为CHWM

3. yDescs是5维向量，dimA = [n,m,e,f,d]，默认存储顺序为NEFDM

v1.4.2版本中存在限制，限制如下：

1. C%32 == 0
2. M%32 == 0
3. 不支持alpha，beta

## swdnnCreate

```C
swdnnStatus_t swdnnCreate(swdnnHandle_t *handle)
```

**功能描述**：

创建句柄，初始化资源，分配结构体空间

**参数描述**：

-	handle：输入/输出，控制句柄

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_ALLOC_FAILED分配内存失败

## swdnnCreateActivationDescriptor

```C
swdnnStatus_t swdnnCreateActivationDescriptor(swdnnActivationDescriptor_t
	*activationDesc)
```

**功能描述**：

创建激活描述符，分配结构体空间

**参数描述**：

-	activationDesc：输入/输出

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

## swdnnCreateAttnDescriptor

```C
swdnnCreateAttnDescriptor(swdnnAttnDescriptor_t *attnDesc)
```

**功能描述**：

创建描述符，分配结构体空间

**参数描述**：

-	attnDesc：输入/输出

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

## swdnnCreateConvolutionDescriptor

```C
swdnnStatus_t swdnnCreateConvolutionDescriptor(swdnnConvolutionDescriptor_t
	*convDesc)
```

**功能描述**：

创建卷积描述符，分配结构体空间

**参数描述**：

-	convDesc：输入/输出

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

## swdnnCreateDropoutDescriptor

```C
swdnnStatus_t swdnnCreateDropoutDescriptor(swdnnDropoutDescriptor_t
	*dropoutDesc)
```

**功能描述**：

创建Dropout描述符，分配结构体空间

**参数描述**：

-	dropoutDesc：输入/输出

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

## swdnnCreateEmbeddingDescriptor

```C
swdnnStatus_t swdnnCreateEmbeddingDescriptor(swdnnEmbeddingDescriptor_t
	*EmbeddingDesc)
```

**功能描述**：

创建Embedding描述符，分配结构体空间

**参数描述**：

-	EmbeddingDesc：输入/输出

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

## swdnnCreateFilterDescriptor

```C
swdnnStatus_t swdnnCreateFilterDescriptor(swdnnFilterDescriptor_t
	*filterDesc)
```

**功能描述**：

创建卷积核描述符，分配结构体空间

**参数描述**：

-	filterDesc：输入/输出

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

## swdnnCreateFusedOpsConstParamPack

```C
swdnnCreateFusedOpsConstParamPack( 
	swdnnFusedOpsConstParamPack_t *constPack, 
	swdnnFusedOps_t ops);
```

**功能描述**：

创建FusedOpConstParam的描述符

**参数描述**：

-   constPack：输出，创建的FusedOpConstParam描述符
-   ops：输入，设置的FusedOp模式

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnCreateFusedOpsPlan

```C
swdnnCreateFusedOpsPlan( 
	swdnnFusedOpsPlan_t *plan, 
	swdnnFusedOps_t ops)
```

**功能描述**：

创建FusedOpsPlan描述符

**参数描述**：

-   plan：输出，创建的FusedOpsPlan描述符
-   ops：输入，设置的FusedOp操作

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnCreateFusedOpsVariantParamPack

```C
swdnnCreateFusedOpsVariantParamPack( 
	swdnnFusedOpsVariantParamPack_t *varPack, 
	swdnnFusedOps_t ops)
```

**功能描述**：

创建FusedOpsVariantParam描述符

**参数描述**：

-   varPack：输出，需要创建的FusedOpsVariantParam描述符
-   ops：输入，设置的fusedOp操作

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnCreateOpTensorDescriptor

```C
swdnnCreateOpTensorDescriptor( 
	swdnnOpTensorDescriptor_t *opTensorDesc);
```

**功能描述**：

创建Op操作描述符

**参数描述**：

opTensorDesc：输出，需要被创建的Op描述符

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnCreatePoolingDescriptor

```C
swdnnStatus_t swdnnCreatePoolingDescriptor(swdnnPoolingDescriptor_t
	*poolingDesc)
```

**功能描述**：

创建池化描述符，分配结构体空间

**参数描述**：

-	poolingDesc：输入/输出

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

## swdnnCreateReduceTensorDescriptor

```C
swdnnCreateReduceTensorDescriptor( 
	swdnnReduceTensorDescriptor_t *reduceTensorDesc);
```

**功能描述**：

创建TensorTransform的描述符。

**参数描述**：

-   transformDesc：创建的TensorTransform的描述符

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnCreateRNNDescriptor

```C
swdnnCreateRNNDescriptor(swdnnRNNDescriptor_t *rnnDesc)
```

**功能描述**：

创建RNN描述符，分配结构体空间

**参数描述：**

-	rnnDesc：输入/输出

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

## swdnnCreateSeqDataDescriptor

```C
swdnnCreateSeqDataDescriptor(swdnnSeqDataDescriptor_t *seqDataDesc)
```

**功能描述**：

创建SeqData描述符，分配结构体空间

**参数描述：**

-	seqDataDesc：输入/输出

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

## swdnnCreateTensorDescriptor

```C
swdnnStatus_t swdnnCreateTensorDescriptor(swdnnTensorDescriptor_t
	*tensorDesc)
```

**功能描述**：

创建张量描述符，分配结构体空间

**参数描述**：

- tensorDesc：输入/输出

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

## swdnnCreateTensorTransformDescriptor

```C
swdnnCreateTensorTransformDescriptor(swdnnTensorTransformDescriptor_t
	*transformDesc);
```

**功能描述**：

创建TensorTransform的描述符。

**参数描述：**

-	transformDesc：创建的TensorTransform的描述符

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnDestroy

```C
swdnnStatus_t swdnnDestroy(swdnnHandle_t handle)
```

**功能描述**：

释放句柄，释放资源，释放结构体空间

**参数描述：**

-	handle：输入

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

## swdnnDestroyActivationDescriptor

```C
swdnnStatus_t swdnnDestroyActivationDescriptor(swdnnActivationDescriptor_t
	activationDesc)
```

**功能描述**：

释放激活描述符，释放结构体空间

**参数描述**：

-	activationDesc：输入

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnDestoryAttnDrscriptor

```C
swdnnDestroyAttnDescriptor(swdnnAttnDescriptor_t attnDesc)
```

**功能描述**：

释放描述符，释放结构体空间

**参数描述**：

-	attnDesc：输入

**返回值**：

1. SWDNN_STATUS_SUCCESS：成功

## swdnnDestroyConvolutionDescriptor

```C
swdnnStatus_t swdnnDestroyConvolutionDescriptor(swdnnConvolutionDescriptor_t convDesc)
```

**功能描述**：

释放卷积描述符，释放结构体空间

**参数描述**：

-	convDesc：输入

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnDestroyDropoutDescriptor

```C
swdnnStatus_t swdnnDestroyDropoutDescriptor(swdnnDropoutDescriptor_t
	dropoutDesc)
```

**功能描述**：

释放Dropout描述符，释放结构体空间

**参数描述**：

-	dropoutDesc：输入

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnDestroyEmbeddingDescriptor

```C
swdnnStatus_t swdnnDestroyEmbeddingDescriptor(swdnnEmbeddingDescriptor_t
	EmbeddingDesc)
```

**功能描述**：

释放Embedding描述符，释放结构体空间

**参数描述**：

-	EmbeddingDesc：输入

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnDestroyFilterDescriptor

```C
swdnnStatus_t swdnnDestroyFilterDescriptor(swdnnFilterDescriptor_t
	filterDesc)
```

**功能描述**：

销毁卷积核描述符，释放结构体空间

**参数描述：**

-	filterDesc：输入

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

## swdnnDestroyFusedOpsConstParamPack

```C
swdnnDestroyFusedOpsConstParamPack( 
	swdnnFusedOpsConstParamPack_t constPack)
```

**功能描述**：

释放FusedOpConstParam描述符

**参数描述**：

-   constPack：输入，需要释放的FusedOpConstParam描述符

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnDestroyFusedOpsPlan

```C
swdnnDestroyFusedOpsPlan( 
	swdnnFusedOpsPlan_t plan)
```

**功能描述**：

释放FusedOpsPlan描述符

**参数描述**：

plan：输入，释放的FusedOpsPlan描述符

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnDestroyFusedOpsVariantParamPack

```C
swdnnDestroyFusedOpsVariantParamPack( 
    swdnnFusedOpsVariantParamPack_t varPack)
```

**功能描述**：

释放FusedOpsVariantParam描述符

**参数描述**：

-   varPack：输入，需要释放的FusedOpsVariantParam描述符

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnDestroyOpTensorDescriptor

```C
swdnnDestroyOpTensorDescriptor( 
	swdnnOpTensorDescriptor_t opTensorDesc);
```

**功能描述**：

释放OpTensor描述符

**参数描述**：

-   opTensorDesc：输入，需要释放的Op描述符

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnDestroyPoolingDescriptor

```C
swdnnStatus_t swdnnDestroyPoolingDescriptor(swdnnPoolingDescriptor_t
	poolingDesc)
```

**功能描述**：

释放池化描述符，释放结构体空间

**参数描述**：

-	poolingDesc：输入

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnDestroyReduceTensorDescriptor

```C
swdnnDestroyReduceTensorDescriptor( 
	swdnnReduceTensorDescriptor_t reduceTensorDesc)
```

**功能描述**：

释放ReduceTensor描述符。

**参数描述**：

-   reduceTensorDesc: 输入，要被释放的ReduceTensor描述符。

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnDestoryRNNDescriptor

```C
swdnnDestroyRNNDescriptor(swdnnRNNDescriptor_t rnnDesc)
```

**功能描述**：

销毁RNN描述符，释放结构体空间

**参数描述：**

-	rnnDesc：输入

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

## swdnnDestorySeqDataDescriptor

```C
swdnnDestroySeqDataDescriptor(swdnnSeqDataDescriptor_t seqDataDesc)
```

**功能描述**：

释放SeqData描述符，释放结构体空间

**参数描述：**

-	seqDataDesc：输入

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnDestroyTensorDescriptor

```C
swdnnStatus_t swdnnDestroyTensorDescriptor(swdnnTensorDescriptor_t
	tensorDesc)
```

**功能描述**：

释放张量描述符，释放结构体空间

**参数描述**：

-	tensorDesc：输入

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnDestroyTensorTransformDescriptor

```C
swdnnDestroyTensorTransformDescriptor(swdnnTensorTransformDescriptor_t
	transformDesc);
```

**功能描述**：

销毁TensorTransform描述符，释放结构体空间

**参数描述：**

-	transformDesc：输入，TensorTransform描述符。

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnDropoutBackward

```C
swdnnStatus_t swdnnDropoutBackward( 
	swdnnHandle_t handle, 
	const swdnnDropoutDescriptor_t dropoutDesc, 
	const swdnnTensorDescriptor_t dydesc, 
	const void *dy, 
	const swdnnTensorDescriptor_t dxdesc, 
	void *dx, 
	void *reserveSpace, 
	size_t reserveSpaceSizeInBytes)
```

**功能描述**：

反向过程中的Dropout计算

**参数描述**：

-	handle：输入，控制句柄 
-	dropoutDesc：输入，dropout的描述符 
-	alpha, beta：输入，扩展因子 
-	dyDesc：输入，dy的张量描述符 
-	dy：输入，dy的首地址指针 
-	dxDesc：输入，dx的张量描述符 
-	dx：输出，dx的首地址指针 
-	reserveSpace：输入，Dropout正向过程中的保存空间首地址 
-	reserveSpaceSizeInBytes，输入，Dropout正向过程中的保存空间大小

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

## swdnnDropoutForward

```C
swdnnStatus_t swdnnDropoutForward( 
	swdnnHandle_t handle, 
	const swdnnDropoutDescriptor_t dropoutDesc, 
	const swdnnTensorDescriptor_t xdesc, 
	const void *x, 
	const swdnnTensorDescriptor_t ydesc, 
	void *y, 
	void *reserveSpace, 
	size_t reserveSpaceSizeInBytes)
```

**功能描述**：

正向过程中的Dropout计算

**参数描述**：

-	handle：输入，控制句柄 
-	dropoutDesc：输入，dropout的描述符 
-	alpha, beta：输入，扩展因子 
-	xDesc：输入，x的张量描述符 
-	x：输入，x的首地址指针 
-	yDesc：输入，y的张量描述符 
-	y：输出，y的首地址指针 
-	reserveSpace：输入，Dropout正向过程中的保存空间首地址 
-	reserveSpaceSizeInBytes，输入，Dropout正向过程中的保存空间大小

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

## swdnnDropoutGetReserveSpaceSize

```C
swdnnStatus_t swdnnDropoutGetReserveSpaceSize(swdnnTensorDescriptor_t
	xDesc,size_t *sizeInBytes)
```

**功能描述**：

获取Dropout的保留空间大小

**参数描述**：

-	xDesc：输入，x的张量描述符
-	sizeInBytes，输出，Dropout过程中的保存空间大小

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

## swdnnDropoutGetStatesSize

```C
swdnnStatus_t swdnnDropoutGetStatesSize(swdnnHandle_t handle,size_t
	*sizeInBytes)
```

**功能描述**：

获得dropout所需的states空间大小

**参数描述**：

-	handle：输入，控制句柄

sizeInBytes，输出，States的空间大小

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

**备注：**

1. 实际计算中，直接为reserveSpace分配输入数据同样大小

## swdnnEmbeddingForward

```C
swdnnStatus_t swdnnEmbeddingForward( 
	swdnnHandle_t swdnnHandle, 
	swdnnEmbeddingDesciptor_t embeddingDesc, 
	swdnnSeqDataDescriptor_t indicesDesc, 
	const void* indices, 
	swdnnSeqDataDescriptor_t weightDesc, 
	const void* weight, 
	const void*beta, 
	swdnnSeqDataDescriptor_t outDesc, 
	const void* outData);
```

**功能描述**：

Embedding前向

**参数描述**：

handle：输入，控制句柄

embeddingDesc，输入，embedding描述符

indicesDesc，输入，indicesDesc描述符

indices，输入，indices首地址指针

weightDesc，输入，weights描述符

weight，输入，weight的首地址指针

outDesc，输入，outDesc描述符

outData，输出，outData首地址

**返回值**：

1.  SWDNN_STATUS_SUCCESS成功

**备注：**

1. 支持float和half计算

## swdnnEmbeddingBackward

```C
swdnnStatus_t swdnnEmbeddingBackward( 
	swdnnHandle_t swdnnHandle, 
	swdnnEmbeddingDesciptor_t embeddingDesc, 
	swdnnSeqDataDescriptor_t doutDesc 
	const void* dout 
	swdnnSeqDataDescriptor_t indicesDesc 
	const void* indices 
	const void*beta 
	swdnnSeqDataDescriptor_t dwDesc, 
	const void* dweights,);
```

**功能描述**：

Embedding反向

**参数描述：**

-	handle：输入，控制句柄 
-	embeddingDesc，输入，embedding描述符 
-	doutDesc，输入，doutDesc描述符 
-	dout，输入 
-	indicesDesc，输入，indicesDesc描述符 
-	indices，输入 
-	dwDesc，输入，dweights描述符 
-	dweights，输出，dweights首地址

**返回值：**

1.  SWDNN_STATUS_SUCCESS成功

**备注：**

1. 支持float和half计算

## swdnnFusedOpsExecute

```C
swdnnFusedOpsExecute( 
	swdnnHandle_t handle, 
	const swdnnFusedOpsPlan_t plan, 
	swdnnFusedOpsVariantParamPack_t varPack);
```

**功能描述**：

进行FusedOp计算

**参数描述**：

-   handle：输入，设备句柄
-   plan：输入，FusedOpsPlan描述符
-   varPack：输入/输出，FusedOpsVariantParam描述符

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_BAD_PARAM 参数错误

**note**：

1. fusedOp操作当前版本并未完全支持所有参数，详细信息查看swdnnFusedOps_t。

## swdnnGetActivationDescriptor

```C
swdnnStatus_t swdnnGetActivationDescriptor( 
	const swdnnActivationDescriptor_t activationDesc, 
	swdnnActivationMode_t *mode, 
	swdnnNanPropagation_t *reluNanOpt, 
	float *coef)
```

**功能描述**：

获取激活描述符的参数值

**参数描述**：

-	activationDesc：输入，激活描述符 
-	mode：输出，激活模式 
-	reluNanOpt：输出，Nan传播模式 
-	coef：输出，系数，在RELU中代表C值，在ELU中代表alpha值。

**返回值**：

1.  SWDNN_STATUS_SUCCESS成功

## swdnnGetAttnDescroptor

```C
swdnnGetAttnDescriptor(swdnnAttnDescriptor_t attnDesc, 
	unsigned *attnMode, 
	int *nHeads, 
	double *smScaler, 
	swdnnDataType_t *dataType, 
	swdnnDataType_t *computePrec, 
	swdnnMathType_t *mathType, 
	swdnnDropoutDescriptor_t *attnDropoutDesc, 
	swdnnDropoutDescriptor_t *postDropoutDesc, 
	int *qSize, 
	int *kSize, 
	int *vSize, 
	int *qProjSize, 
	int *kProjSize, 
	int *vProjSize, 
	int *oProjSize, 
	int *qoMaxSeqLength, 
	int *kvMaxSeqLength, 
	int *maxBatchSize, 
	int *maxBeamSize);
```

**功能描述**：

获取神经元函数描述符的参数值

**参数描述**：

-	attnMode：输入，神经元函数模式 
-	int *nHeads：输出，nHeads首地址指针 
-	smScaler：输出，smScaler首地址指针 
-	dataType：输出，attn数据类型 
-	computePrec：输出，computePrec数据类型 
-	mathType：输出，计算类型 
-	attnDropoutDesc：输入，attnDropout描述符 
-	postDropoutDesc：输入，postDropout描述符 
-	qSize：输出，qSize首地址指针 
-	kSize：输出，kSize首地址指针 
-	vSize：输出，kSize首地址指针 
-	qProjSize：输出，qProjSize首地址指针 
-	kProjSize：输出，kProjSize首地址指针 
-	vProjSize：输出，vProjSize首地址指针 
-	oProjSize：输出，oProjSize首地址指针 
-	qoMaxSeqLength：输出，qoMaxSeqLength首地址指针 
-	kvMaxSeqLength：输出，kvMaxSeqLenth首地址指针 
-	maxBatchSize：输出，maxBatchSize首地址指针 
-	maxBeamSize：输出，maxBeanSize首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetBiasMode

```C
swdnnGetBiasMode(const swdnnRNNDescriptor_t rnnDesc,swdnnRNNBiasMode_t *biasMode)
```

**功能描述**：

获取偏置模式

**参数描述：**

-	rnnDesc：输入，rnn描述符 
-	biasMode：输出，偏置模式

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

**备注：**

1. 目前不支持BIAS计算

## swdnnGetConvolution2dDescriptor

```C
swdnnStatus_t swdnnGetConvolution2dDescriptor( 
	const swdnnConvolutionDescriptor_t convDesc, 
	int *pad_h, 
	int *pad_w, 
	int *u, 
	int *v, 
	int *dilation_h, 
	int *dilation_w, 
	swdnnConvolutionMode_t *mode, 
	swdnnDataType_t *DataType)
```

**功能描述**：

获取二维卷积描述符的参数值

**参数描述**：

-	convDesc：输入，卷积描述符 
-	pad_h：输出，h方向的padding值地址 
-	pad_w：输出，w方向的padding值地址 
-	u：输出，垂直方向的卷积核跨步地址 
-	v：输出，水平方向的卷积核跨步地址 
-	dilation_h：输出，h方向的dilation值地址 
-	dilation_w：输出，w方向的dilation值地址 
-	mode：输出，卷积模式 
-	DataType：输出，卷积数据类型

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetConvolution2dForwardOutputDim

```C
swdnnGetConvolution2dForwardOutputDim(const swdnnConvolutionDescriptor_t convDesc, 
	const swdnnTensorDescriptor_t inputTensorDesc, 
	const swdnnFilterDescriptor_t filterDesc, 
	int *n, 
	int *c, 
	int *h, 
	int *w);
```

**功能描述**：

获取卷积前向输出维度参数值

**参数描述：**

-	convDesc：输入，卷积描述符 
-	inputTensorDesc：输入，input张量描述符 
-	n：输出，卷积核个数 
-	c：输出，通道数 
-	h：输出，高度 
-	w：输出，宽度

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetConvolutionBackwardDataWorkspaceSize

```C
swdnnGetConvolutionBackwardDataWorkspaceSize( 
	swdnnHandle_t handle, 
	const swdnnFilterDescriptor_t wDesc, 
	const swdnnTensorDescriptor_t dyDesc, 
	const swdnnConvolutionDescriptor_t convDesc, 
	const swdnnTensorDescriptor_t dxDesc, 
	swdnnConvolutionBwdDataAlgo_t algo, 
	size_t *sizeInBytes)
```

**功能描述**：

获取反向过程中数据残差计算的工作空间大小

**参数描述：**

-	handle：输入，控制句柄 
-	wDesc：输入，w的卷积核描述符 
-	dyDesc：输入，dy的张量描述符 
-	convDesc：输入，卷积描述符 
-	dxDesc：输入，dx的张量描述符 
-	algo：输入，卷积反向数据残差计算的算法描述符 
-	sizeInBytes：输出，工作空间大小

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetConvolutionBackwardFilterWorkspaceSize

```C
swdnnGetConvolutionBackwardFilterWorkspaceSize( 
	swdnnHandle_t handle, 
	const swdnnTensorDescriptor_t xDesc, 
	const swdnnTensorDescriptor_t dyDesc, 
	const swdnnConvolutionDescriptor_t convDesc, 
	const swdnnFilterDescriptor_t dwDesc, 
	swdnnConvolutionBwdFilterAlgo_t algo, 
	size_t *sizeInBytes);
```

**功能描述**：

获取卷积反向卷积核更新值计算工作空间的大小

**参数描述：**

-	handle：输入，控制句柄 
-	xDesc：输入，x的张量描述符 
-	dyDesc：输入，dy的张量描述符 
-	convDesc：输入，卷积描述符 
-	dwDesc：输入，dw的卷积核描述符 
-	algo：输入，算法描述符 
-	sizeInBytes：输出，工作空间大小

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetConvolutionForwardWorkspaceSize

```C
swdnnGetConvolutionForwardWorkspaceSize( 
	swdnnHandle_t handle, 
	const swdnnTensorDescriptor_t xDesc, 
	const swdnnFilterDescriptor_t wDesc, 
	const swdnnConvolutionDescriptor_t convDesc, 
	const swdnnTensorDescriptor_t yDesc, 
	swdnnConvolutionFwdAlgo_t algo, 
	size_t *sizeInBytes);
```

**功能描述**：

获取卷积正向卷积计算工作空间的大小

**参数描述：**

-	handle：输入，控制句柄 
-	xDesc：输入，x的张量描述符 
-	yDesc：输入，y的张量描述符 
-	convDesc：输入，卷积描述符 
-	wDesc：输入，w的卷积核描述符 
-	algo：输入，算法描述符 
-	sizeInBytes：输出，工作空间大小

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetConvolutionGroupCount

```C
swdnnStatus_t swdnnGetConvolutionGroupCount( 
	swdnnConvolutionDescriptor_t convDesc, 
	int *groupCount)
```

**功能描述**：

获取卷积的分组参数值

**参数描述**：

-	convDesc：输入，卷积描述符 
-	groupCount：输出，分组值地址

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetConvolutionMathType

```C
swdnnStatus_t swdnnGetConvolutionMathType( 
	swdnnConvolutionDescriptor_t convDesc, 
	swdnnMathType_t *mathType)
```

**功能描述**：

获取卷积的数学类型

**参数描述**：

-	convDesc：输入，卷积描述符 
-	mathType：输出，卷积数学类型

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetConvolutionNdDescriptor

```C
swdnnStatus_t swdnnGetConvolutionNdDescriptor( 
	const swdnnConvolutionDescriptor_t convDesc, 
	int arrayLengthRequested, 
	int *arrayLength, 
	int padA[], 
	int filterStrideA[], 
	int dilationA[], 
	swdnnConvolutionMode_t *mode, 
	swdnnDataType_t *dataType)
```

**功能描述**：

获取N维卷积描述符的参数值

**参数描述**：

-	convDesc：输入，卷积描述符 
-	arrayLengthRequested：输入，期望的卷积维度 
-	arrayLength：输出，实际的卷积维度 
-	padA：输出，各个方向的padding值地址 
-	filterStrideA：输出，各个方向的卷积核跨步地址 
-	dilationA：输出，各个方向的dilation值地址 
-	mode：输出，卷积模式 
-	DataType：输出，卷积数据类型

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetDropoutDescriptor

```C
swdnnStatus_t swdnnGetDropoutDescriptor(
	swdnnDropoutDescriptor_t dropoutDesc, 
	swdnnHandle_t handle, 
	float *dropout, 
	void **states, 
	unsigned long long *seed);
```

**功能描述**：

获取Dropout描述符的参数值

**参数描述**：

-	handle：输入，控制句柄 
-	dropoutDesc：输入，dropout描述符 
-	dropout：输出，dropout的首地址指针 
-	states：输出，states的首地址指针 
-	seed：输出，随机种子

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetFilter4dDescriptor

```C
swdnnStatus_t swdnnGetFilter4dDescriptor( 
	const swdnnFilterDescriptor_t filterDesc, 
	swdnnDataType_t *dataType, 
	swdnnTensorFormat_t *format, 
	int *k, 
	int *c, 
	int *h, 
	int *w)
```

**功能描述**：

获取四维卷积核描述符的参数值

**参数描述**：

-	filterDesc：输入，卷积核描述符 
-	dataType：输出，卷积数据类型 
-	format：输出，卷积核格式 
-	k：输出，卷积核个数 
-	c：输出，卷积核通道数 
-	h：输出，卷积核宽度 
-	w：输出，卷积核高度

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetFilterNdDescriptor

```C
swdnnStatus_t swdnnGetFilterNdDescriptor( 
	const swdnnFilterDescriptor_t wDesc, 
	int nbDimsRequested, 
	swdnnDataType_t *dataType, 
	swdnnTensorFormat_t *format, 
	int *nbDims, 
	int filterDimA[])
```

**功能描述**：

获取N维卷积核描述符的参数值

**参数描述**：

-	wDesc：输入，卷积核描述符 
-	nbDimsRequested：输入，期望的卷积核维度 
-	dataType：输出，卷积核数据类型 
-	format：输出，卷积核格式 
-	nbDims：输出，实际的卷积核维度 
-	filterDimA：输出，各个方向的卷积核维度值地址

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetFusedOpsConstParamPackAttribute

```C
swdnnGetFusedOpsConstParamPackAttribute(const
	swdnnFusedOpsConstParamPack_t constPack, 
	swdnnFusedOpsConstParamLabel_t paramLabel, 
	void *param, 
	int *isNULL);
```

**功能描述**：

获取FusedOpConstParam参数

**参数描述**：

-   constPack：输入，获取的FusedOpConstParam描述符
-   paramLabel：输出，设置的参数标签
-   param：输出，设置的参数
-   isNULL：出入，参数是否为NULL

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetFusedOpsVariantParamPackAttribute

```C
swdnnGetFusedOpsVariantParamPackAttribute( 
	const swdnnFusedOpsVariantParamPack_t varPack, 
	swdnnFusedOpsVariantParamLabel_t paramLabel, 
	void *ptr)
```

**功能描述**：

获取FusedOpsVariantParam描述符设置

**参数描述**：

-   varPack：输入，需要获取的FusedOpsVariantParam描述符
-   paramLabel：输出，获取的设置标签
-   ptr：输出，获取的设置

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetMultiHeadAttnBuffers

```C
swdnnGetMultiHeadAttnBuffers(swdnnHandle_t handle, 
	const swdnnAttnDescriptor_t attnDesc, 
	size_t *weightSizeInBytes, 
	size_t *workSpaceSizeInBytes, 
	size_t *reserveSpaceSizeInBytes)
```

**功能描述：**

获取多头神经元参数值

**参数描述：**

-	handle：输入，控制句柄 
-	attnDesc：输入，attn描述符 
-	weightSizeInBytes：输出，weight大小 
-	workSpaceSizeInBytes：输出，workSpace字节长度 
-	reserveSpaceSizeInBytes：输出，reserveSpace大小

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetMultiHeadAttnWeights

**（暂未支持）**

```C
swdnnGetMultiHeadAttnWeights(swdnnHandle_t handle, 
	const swdnnAttnDescriptor_t attnDesc, 
	swdnnMultiHeadAttnWeightKind_t wKind, 
	size_t weightSizeInBytes, 
	const void *weights, 
	swdnnTensorDescriptor_t wDesc, 
	void **wAddr)
```

**功能描述**：

获取多头神经元函数权重的参数值

**参数描述**：

-	handle：输入，控制句柄 
-	attnDesc：输入，attn描述符 
-	wKind：输入，多头网络权重类型 
-	weightSizeInBytes：输入，weight大小 
-	weights：输入，weight首地址指针 
-	wDesc：输入，w的张量描述符 
-	wAddr：输出，权重地址

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetOpTensorDescriptor

```C
swdnnGetOpTensorDescriptor(const swdnnOpTensorDescriptor_t opTensorDesc, 
	swdnnOpTensorOp_t *opTensorOp, 
	swdnnDataType_t *opTensorCompType, 
	swdnnNanPropagation_t *opTensorNanOpt);
```

**功能描述**：

获取OpTensor描述符参数

**参数描述**：

-   opTensorDesc：输入，需要获取的Op描述符
-   opTensorOp：输出，获取的Optensor模式
-   opTensorCompType：输出，获取的计算数据类型
-   opTensorNanOpt：输出

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetPooling2dDescriptor

```C
swdnnStatus_t swdnnGetPooling2dDescriptor( 
	const swdnnPoolingDescriptor_t poolingDesc, 
	swdnnPoolingMode_t *mode, 
	swdnnNanPropagation_t *maxpoolingNanOpt, 
	int *windowHeight, 
	int *windowWidth, 
	int *verticalPadding, 
	int *horizontalPadding, 
	int *verticalStride, 
	int *horizontalStride)
```

**功能描述**：

获取二维池化描述符的参数值

**参数描述**：

-	poolingDesc：输入，池化描述符 
-	mode：输出，池化模式值地址 
-	maxpoolingNanOpt：输出，是否传播Nan 
-	windowHeight：输出，池化窗口高度值地址 
-	windowWidth：输出，池化窗口宽度值地址 
-	verticalPadding：输出，垂直Padding值地址 
-	horizontalPadding：输出，水平Padding值地址 
-	verticalStride：输出，垂直跨步值地址 
-	horizontalStride：输出，水平跨步值地址

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetPooling2dForwardOutputDim

```C
swdnnGetPooling2dForwardOutputDim( 
	const swdnnPoolingDescriptor_t poolingDesc, 
	const swdnnTensorDescriptor_t inputTensorDesc, 
	int *n, 
	int *c, 
	int *h, 
	int *w)
```

**功能描述**：

获取二维池化正向输出的维度值

**参数描述**：

-	poolingDesc：输入，池化描述符 
-	inputTensorDesc：输入，输入张量描述符 
-	n：输出，池化输出的数量 
-	c：输出，池化输出的通道数 
-	h：输出，池化输出的高度 
-	w：输出，池化输出的宽度

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetPoolingNdDescriptor

```C
swdnnStatus_t swdnnGetPoolingNdDescriptor( 
	const swdnnPoolingDescriptor_t poolingDesc, 
	int nbDimsRequested, 
	swdnnPoolingMode_t *mode, 
	swdnnNanPropagation_t *maxpoolingNanOpt, 
	int *nbDims, 
	int windowDimA[], 
	int paddingA[], 
	int strideA[])
```

**功能描述**：

获取N维池化描述符的参数值

**参数描述**：

-	poolingDesc：输入，池化描述符 
-	nbDimsRequested：输入，期望的维度值 
-	mode：输出，池化模式值地址 
-	maxpoolingNanOpt：输出，是否传播Nan 
-	nbDims：输出，实际的维度值地址 
-	windowDimA：输出，池化窗口各个维度值地址 
-	paddingA：输出，池化各个维度Padding值地址 
-	strideA：输出，池化各个维度跨步值地址

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetReduceTensorDescriptor

```C
swdnnGetReduceTensorDescriptor(const swdnnReduceTensorDescriptor_t
	reduceTensorDesc, 
	swdnnReduceTensorOp_t *reduceTensorOp, 
	swdnnDataType_t *reduceTensorCompType, 
	swdnnNanPropagation_t *reduceTensorNanOpt, 
	swdnnReduceTensorIndices_t *reduceTensorIndices, 
	swdnnIndicesType_t *reduceTensorIndicesType);
```

**功能描述**：

获取ReduceTensor描述符的值

**参数描述**：

-   reduceTensorDesc: 输入，要被获取的ReduceTensor描述符。
-   reduceTensorOp: 输出，ReduceTensor的计算模式
-   reduceTensorCompType: 输出，ReduceTensor计算的数据类型
-   reduceTensorNanOpt: 输出
-   reduceTensorIndices: 输出，ReduceTensor的indices操作类型
-   reduceTensorIndicesType: 输出，indices的返回类型

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetReduceTensorWorkspaceSize

```C
swdnnGetReductionWorkspaceSize(swdnnHandle_t handle, 
	const swdnnReduceTensorDescriptor_t reduceTensorDesc, 
	const swdnnTensorDescriptor_t aDesc, 
	const swdnnTensorDescriptor_t cDesc, 
	size_t *sizeInBytes);
```

**功能描述**：

获得reduce操作需要的空间大小。

**参数描述**：

-   handle：输入，设备句柄
-   reduceTensorDesc：输入，reduce操作描述符
-   aDesc：输入，需要被reduce的数据描述符
-   cDesc：输入，reduce的结果数据描述符
-   sizeInBytes：输出，需要的空间大小

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetReductionIndicesSize

```C
swdnnGetReductionIndicesSize(swdnnHandle_t handle, 
	const swdnnReduceTensorDescriptor_t reduceTensorDesc, 
	const swdnnTensorDescriptor_t aDesc, 
	const swdnnTensorDescriptor_t cDesc, 
	size_t *sizeInBytes);
```

**功能描述**：

获得indices需要的空间大小。

**参数描述**：

-   handle：输入，设备句柄
-   reduceTensorDesc：输入，reduce操作描述符
-   aDesc：输入，需要被reduce的数据描述符
-   cDesc：输入，reduce的结果数据描述符
-   sizeInBytes：输出，需要的空间大小

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

**note**：

1. 目前reduce不支持indice操作

## swdnnGetRNNDescriptor

```C
swdnnGetRNNDescriptor(swdnnHandle_t handle, 
	swdnnRNNDescriptor_t rnnDesc, 
	int *hiddenSize, 
	int *numLayers, 
	swdnnDropoutDescriptor_t *dropoutDesc, 
	swdnnRNNInputMode_t *inputMode, 
	swdnnDirectionMode_t *direction, 
	swdnnRNNMode_t *mode, 
	swdnnRNNAlgo_t *algo, 
	swdnnDataType_t *mathPrec)
```

**功能描述**：

获取RNN描述符的参数值

**参数描述：**

-	rnnDesc：输入，rnn描述符 
-	hiddenSize: 输入，隐藏层大小 
-	numLayers：输出，网络层数 
-	dropoutDesc：输出，dropout描述符 
-	inputMode：输出，rnn第一层操作 
-	direction：输出，循环方向 
-	mode：输出，网络类型 
-	algo：rnn支持的算法类型 
-	mathPrec：输出，计算模式

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetRNNWorkspaceSize

```C
swdnnGetRNNWorkspaceSize(swdnnHandle_t handle, 
	const swdnnRNNDescriptor_t rnnDesc, 
	const int seqLength, 
	const swdnnTensorDescriptor_t *xDesc, 
	size_t *sizeInBytes)
```

**功能描述**：

获取RNN工作空间大小

**参数描述：**

-	handle：输入，控制句柄 
-	rnnDesc：输入，rnn描述符 
-	seqLength：输入，时间序列长度 
-	xDesc：输入，x的张量描述符 
-	sizeInBytes：输出，字节长度

**返回值：**

1.  SWDNN_STATUS_SUCCESS成功

## swdnnGetRNNTrainingReserveSize

```C
swdnnGetRNNTrainingReserveSize(swdnnHandle_t handle, 
	const swdnnRNNDescriptor_t rnnDesc, 
	const int seqLength, 
	const swdnnTensorDescriptor_t *xDesc, 
	size_t *sizeInBytes);
```

**功能描述**：

获取RNN训练reserve空间大小

**参数描述：**

-	handle：输入，控制句柄 
-	rnnDesc：输入，rnn描述符 
-	seqLength：输入，时间序列长度 
-	xDesc：输入，x的张量描述符 
-	sizeInBytes：输出，字节长度

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetRNNParamsSize

```C
swdnnGetRNNParamsSize(swdnnHandle_t handle, 
	const swdnnRNNDescriptor_t rnnDesc, 
	const swdnnTensorDescriptor_t xDesc, 
	size_t *sizeInBytes, 
	swdnnDataType_t dataType);
```

**功能描述**：

获取RNN共享层size

**参数描述：**

-	handle：输入，控制句柄 
-	rnnDesc：输入，rnn描述符 
-	dataType：输入，数据类型 
-	xDesc：输出，x的张量描述符 
-	sizeInBytes：输出，字节长度

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetSeqDataDescriptor

```C
swdnnGetSeqDataDescriptor(const swdnnSeqDataDescriptor_t seqDataDesc, 
	swdnnDataType_t *dataType, 
	int *nbDims, 
	int nbDimsRequested, 
	int dimA[], 
	swdnnSeqDataAxis_t axes[], 
	size_t *seqLengthArraySize, 
	size_t seqLengthSizeRequested, 
	int seqLengthArray[], 
	void *paddingFill);
```

**功能描述**：

获取SeqData描述符的参数值

**参数描述**：

-	SeqDataDesc：输入，SeqData描述符 
-	dataType：输出，数据类型 
-	nbDims：输出，实际的维度值地址 
-	nbDimsRequested：输入，输出的维度长度 
-	dimA：各个方向维度值地址 
-	nbDimsRequested：输入，期望的维度值 
-	axes：输入，axes描述符 
-	seqLengthArraySize：输入，时间序列向量实际参数值 
-	seqLengthSizeRequested：输入，输出的时间序列期长度 
-	seqLengthArray：输入，seqLength向量 
-	paddingFill：输入，填充向量

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetTensor4dDescriptor

```C
swdnnStatus_t swdnnGetTensor4dDescriptor( 
	const swdnnTensorDescriptor_t tensorDesc, 
	swdnnDataType_t *dataType, 
	int *n, 
	int *c, 
	int *h, 
	int *w, 
	int *nStride, 
	int *cStride, 
	int *hStride, 
	int *wStride)
```

**功能描述**：

获取四维张量描述符的参数值

**参数描述**：

-	tensorDesc：输入，张量描述符 
-	dataType：输出，张量的数据类型 
-	n：输出，张量的个数值地址 
-	c：输出，张量的通道数值地址 
-	h：输出，张量的高度值地址 
-	w：输出，张量的宽度值地址 
-	nStride：输出，张量的个数跨步值地址 
-	cStride：输出，张量的通道跨步值地址 
-	hStride：输出，张量的高度跨步值地址 
-	wStride：输出，张量的宽度跨步值地址

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetTensorNdDescriptor

```C
swdnnStatus_t swdnnGetTensorNdDescriptor( 
	const swdnnTensorDescriptor_t tensorDesc, 
	int nbDimsRequested, 
	swdnnDataType_t *dataType, 
	int *nbDims, 
	int dimA[], 
	int strideA[])
```

**功能描述**：

获取N维张量描述符的参数值

**参数描述**：

-	tensorDesc：输入，张量描述符 
-	nbDimsRequested：输入，期望的张量维度 
-	dataType：输出，张量的数据类型值地址 
-	nbDims：输出，实际的张量维度值地址 
-	dimA：输出，各个方向张量的维度值地址 
-	strideA：输出，各个方向张量的跨步值地址

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetTensorSizeInBytes

**（暂不支持）**

```C
swdnnStatus_t swdnnGetTensorSizeInBytes(const swdnnTensorDescriptor_t tensorDesc,size_t *size)
```

**功能描述**：

获取张量所占字节数

**参数描述**：

-	tensorDesc：输入，张量描述符 
-	size：输出，张量在内存中所占空间大小

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetTensorTransformDescriptor

```C
swdnnGetTensorTransformDescriptor( 
	swdnnTensorTransformDescriptor_t transformDesc, 
	unsigned int nbDimsRequested, 
	swdnnTensorFormat_t *destFormat, 
	int padBeforeA[], 
	int padAfterA[], 
	unsigned int foldA[], 
	swdnnFoldingDirection_t *direction);
```

**功能描述**：

获取TensorTransform描述符的各成员的值。

**参数描述：**

-	transformDesc：输入， Transform张量描述符。 
-	nbDimsRequested：输入,，padBeforeA、padAfterA数组维度 
-	destFormat：输出，目标张量的格式 
-	padBeforeA[]：输入，各个方向padBrfore的参数值 
-	padAfterA[]：输入，各个方向padAfter的参数值 
-	foldA[]：输入，fold数组 
-	direction：输出, fold模式

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnGetVersion

```C
size_t swdnnGetVersion()
```

**功能描述**：

获取算法库版本

**返回值**：

算法库版本号

## swdnnGumbelSoftmaxBackward

```C
swdnnGumbelSoftmaxBackward( 
	swdnnHandle_t handle, 
	swdnnSoftmaxMode_t mode, 
	unsigned oneHot, 
	double tau, 
	const void *alpha, 
	const swdnnTensorDescriptor_t xDesc, 
	const void *x, 
	const swdnnTensorDescriptor_t yDesc, 
	const void *y, 
	const swdnnTensorDescriptor_t dyDesc, 
	const void *dy, 
	const void *beta, 
	const swdnnTensorDescriptor_t dxDesc, 
	void *dx, 
	void *reserveSpace);
```

**功能描述**：

反向过程中gumbelsoftmax函数

**参数描述**：

-   handle：输入，控制句柄
-   mode：输入，softmax模式
-   oheHot：输入
-   tau：输入
-   alpha, beta：输入，扩展因子
-   xDesc：输入，x的张量描述符
-   x：输入，x的首地址指针
-   yDesc：输入，y的张量描述符
-   y：输入，y的首地址指针
-   dyDesc：输入，dy的张量描述符
-   dy：输入，y的首地址指针
-   dxDesc：输入，dx的张量描述符
-   dx：输出，dx的首地址指针
-   reserveSpace：输出

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1. SWDNN_DATA_FLOAT

2. SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1. SWDNN_TENSOR_NHWC

**目前已支持的softmax算法swdnnSoftmaxAlgorithm_t**：

1. SWDNN_SOFTMAX_ACCURATE

**目前已支持的softmax运算模式swdnnSoftmaxMode_t：**

1. SWDNN_SOFTMAX_MODE_CHANNEL

2. SWDNN_SOFTMAX_MODE_INSTANCE

## swdnnGumbelSoftmaxForward

```C
swdnnGumbelSoftmaxForward( 
	swdnnHandle_t handle, 
	swdnnSoftmaxMode_t mode, 
	unsigned oneHot, 
	unsigned long long seed, 
	double epsilon, 
	double tau, 
	const void *alpha, 
	const swdnnTensorDescriptor_t xDesc, 
	const void *x, 
	const void *beta, 
	const swdnnTensorDescriptor_t yDesc, 
	void *y, 
	void *reserveSpace);
```

**功能描述**：

正向过程中gumbelsoftmax函数

**参数描述**：

-   handle：输入，控制句柄
-   mode：输入，softmax模式
-   oheHot：输入
-   seed：输入
-   epsilon：输入
-   tau：输入
-   alpha, beta：输入，扩展因子
-   xDesc：输入，x的张量描述符
-   x：输入，x的首地址指针
-   yDesc：输入，y的张量描述符
-   y：输出，y的首地址指针
-   reserveSpace：输出

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1. SWDNN_DATA_FLOAT

2. SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1. SWDNN_TENSOR_NHWC

**目前已支持的softmax算法swdnnSoftmaxAlgorithm_t**：

1. SWDNN_SOFTMAX_ACCURATE

**目前已支持的softmax运算模式swdnnSoftmaxMode_t：**

1. SWDNN_SOFTMAX_MODE_CHANNEL

2. SWDNN_SOFTMAX_MODE_INSTANCE

## swdnnIm2Col

```C
swdnnStatus_t swdnnIm2Col( 
	swdnnHandle_t handle, 
	swdnnTensorDescriptor_t srcDesc, 
	const void *srcData, 
	swdnnFilterDescriptor_t filterDesc, 
	swdnnConvolutionDescriptor_t convDesc, 
	void *colBuffer)
```

**功能描述**：

将src根据权重转换为colBuffer保存模式

**参数描述：**

-	handle：输入，控制句柄 
-	srcDesc：输入， src描述符 
-	srcData: 输出，srcData首地址指针 
-	filterDesc：输入，卷积核描述符 
-	convDesc：输入,卷积描述符 
-	colBuffer：输出

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NCHW

## swdnnInitTransformDest

```C
swdnnInitTransformDest( 
	const swdnnTensorTransformDescriptor_t transformDesc, 
	const swdnnTensorDescriptor_t srcDesc, 
	swdnnTensorDescriptor_t destDesc, 
	size_t *destSizeInBytes);
```

**功能描述**：

初始化和返回TransformTensorEx操作的结果张量的描述符，并返回结果张量的长度。

**参数描述：**

-	transformDesc：输入，Transform张量描述符。 
-	srcDesc：输入，源张量的描述符 
-	destDesc：输入，结果张量的描述符 
-	destSizeInBytes：输出，结果张量的字节长度

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnLayerNormBackward

```C
swdnnLayerNormBackward( 
	swdnnHandle_t swdnnHandle, 
	swdnnLayerMode_t mode, 
	swdnnTensorDescriptor_t dyDesc, 
	const void* dy, 
	swdnnTensorDescriptor_t xDesc, 
	const void*x, 
	swdnnTensorDescriptor_t meanDesc, 
	const void* mean, 
	swdnnTensorDescriptor_t rstdDesc, 
	const void* rstd, 
	swdnnTensorDescriptor_t gemmaDesc, 
	const void* gemma, 
	swdnnTensorDescriptor_t dxDesc, 
	void* dx, 
	swdnnTensorDescriptor_t dgemmaDesc, 
	void* dgemma, 
	swdnnTensorDescriptor_t dbetaDesc, 
	void* dbeta);
```

**功能描述**：

层归一化反向函数

**参数描述：**

-	mode：输入，层归一化模式。 
-	dyDesc：输入，dy的描述符 
-	dy：输入 
-	xDesc：输入，x的描述符 
-	x：输入 
-	meanDesc：输入，mean的描述符 
-	mean：输入 
-	rstdDesc：输入，rstd的描述符 
-	rstd：输入 
-	gammaDesc：输入，gamma的描述符 
-	gamma：输入 
-	dxDesc：输入，dx的描述符 
-	dx：输出 
-	mean：输入，mean的描述符 
-	dgammaDesc：输入，dgamma的描述符 
-	dgamma：输出 
-	dbetaDesc：输入，dbeta的描述符 
-	dbeta：输出

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

**备注：**

1. 支持float和half计算

2. 仅支持NHWC格式

## swdnnLayerNormForward

```C
swdnnLayerNormForward( 
	swdnnHandle_t swdnnHandle, 
	swdnnLayerMode_t mode, 
	swdnnTensorDescriptor_t xDesc, 
	const void* x, 
	swdnnTensorDescriptor_t gammaDesc, 
	const void*gamma, 
	swdnnTensorDescriptor_t betaDesc, 
	const void* beta, 
	double eps, 
	swdnnTensorDescriptor_t yDesc, 
	void* y, 
	swdnnTensorDescriptor_t meanDesc, 
	void* mean, 
	swdnnTensorDescriptor_t rstdDesc, 
	void* rstd);
```

**功能描述**：

层归一化前向函数

**参数描述：**

-	mode：输入：层归一化模式。 
-	xDesc：输入：x的描述符 
-	x：输入 
-	gammaDesc：输入：gamma的描述符 
-	gamma：输入 
-	betaDesc：输入：beta的描述符 
-	beta：输入 
-	yDesc：输出：y的描述符 
-	y：输出 
-	meanDesc：输出：mean的描述符 
-	mean：输出 
-	rstdDesc：输出：rstd的描述符 
-	rstd：输出

**返回值**：

1.  SWDNN_STATUS_SUCCESS成功

**备注：**

1. 支持float和half计算

2. 仅支持NHWC格式

## swdnnLogicalAndTensor

```C
swdnnLogicalAndTensor(
    swdnnHandle_t handle, 
    const swdnnTensorDescriptor_t aDesc,
    const void *A, 
    const swdnnTensorDescriptor_t bDesc,
    const void *B, 
    const swdnnTensorDescriptor_t cDesc,
    void *C);
```
**功能描述**：

张量执行逐元素not计算

**参数描述**：

-	handle：输入，控制句柄 
-	alpha, beta：输入，扩展因子 
-	aDesc：输入，A的张量描述符 
-	A：输入，A的首地址指针 
-	bDesc：输入，B的张量描述符 
-	B：输入，B的首地址指针 
-	cDesc：输入，C的张量描述符 
-	C：输入/输出，C的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_INT32

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC

2.  SWDNN_TENSOR_NCHW

3.  SWDNN_TENSOR_CHWN

## swdnnLogicalNotTensor
```C
swdnnLogicalNotTensor(
    swdnnHandle_t handle, 
    const swdnnTensorDescriptor_t aDesc,
    const void *A, 
    const swdnnTensorDescriptor_t cDesc,
    void *C);
```
**功能描述**：

张量执行逐元素not计算

**参数描述**：

-	handle：输入，控制句柄 
-	alpha, beta：输入，扩展因子 
-	aDesc：输入，A的张量描述符 
-	A：输入，A的首地址指针 
-	cDesc：输入，C的张量描述符 
-	C：输入/输出，C的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_INT32

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC

2.  SWDNN_TENSOR_NCHW

3.  SWDNN_TENSOR_CHWN

## swdnnLogicalOrTensor
```C
swdnnLogicalOrTensor(
    swdnnHandle_t handle, 
    const swdnnTensorDescriptor_t aDesc,
    const void *A, 
    const swdnnTensorDescriptor_t bDesc,
    const void *B, 
    const swdnnTensorDescriptor_t cDesc,
    void *C);
```
**功能描述**：

张量执行逐元素or计算

**参数描述**：

-	handle：输入，控制句柄 
-	alpha, beta：输入，扩展因子 
-	aDesc：输入，A的张量描述符 
-	A：输入，A的首地址指针 
-	bDesc：输入，A的张量描述符 
-	B：输入，A的首地址指针 
-	cDesc：输入，C的张量描述符 
-	C：输入/输出，C的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_INT32

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC

2.  SWDNN_TENSOR_NCHW

3.  SWDNN_TENSOR_CHWN

## swdnnLogicalXorTensor

```C
swdnnLogicalXorTensor(
    swdnnHandle_t handle, const swdnnTensorDescriptor_t aDesc,
    const void *A, 
    const swdnnTensorDescriptor_t bDesc,
    const void *B, 
    const swdnnTensorDescriptor_t cDesc,
    void *C);
```
**功能描述**：

张量执行逐元素xor计算

**参数描述**：

-	handle：输入，控制句柄 
-	alpha, beta：输入，扩展因子 
-	aDesc：输入，A的张量描述符 
-	A：输入，A的首地址指针 
-	bDesc：输入，A的张量描述符 
-	B：输入，A的首地址指针 
-	cDesc：输入，C的张量描述符 
-	C：输入/输出，C的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_INT32

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC

2.  SWDNN_TENSOR_NCHW

3.  SWDNN_TENSOR_CHWN

## swdnnMakeFusedOpsPlan

```C
swdnnMakeFusedOpsPlan(swdnnHandle_t handle, 
	swdnnFusedOpsPlan_t plan, 
	const swdnnFusedOpsConstParamPack_t constPack, 
	size_t *workspaceSizeInBytes);
```

**功能描述**：

设置FusedOpsPlan描述符

**参数描述**：

-   handle：输入，设备句柄
-   plan：输出，设置的FusedOpsPlan描述符
-   constPack：输入，传入的FusedOpsConstParam描述符
-   workspaceSizeInBytes：输出，计算需要的空间大小

**返回值**：

1.  SWDNN_STATUS_SUCCESS成功

## swdnnMultiHeadAttnBackwardWeights

```C
swdnnMultiHeadAttnBackwardWeights(swdnnHandle_t handle, 
	const swdnnAttnDescriptor_t attnDesc, 
	swdnnWgradMode_t addGrad, 
	const swdnnSeqDataDescriptor_t qDesc, 
	const void *queries, 
	const swdnnSeqDataDescriptor_t kDesc, 
	const void *keys, 
	const swdnnSeqDataDescriptor_t vDesc, 
	const void *values, 
	const swdnnSeqDataDescriptor_t doDesc, 
	const void *dout, 
	size_t weightSizeInBytes, 
	const void *weights, 
	void *dweights, 
	size_t workSpaceSizeInBytes, 
	void *workSpace, 
	size_t reserveSpace, 
	void *reserveSpace)
```

**功能描述：**

计算多头注意力反向权值更新过程。

**参数描述：**

-	handle：输入，句柄 
-	attnDesc：输入，attn描述符 
-	qDesc：输入，查询输入queries描述符 
-	queries：输入，查询输入queries的首地址 
-	kDesc：输入，键输入keys的描述符 
-	keys：输入，键输入keys的首地址 
-	vDesc：输入，值输入values的描述符 
-	values：输入，值输入values的首地址 
-	doDesc：输入，多头注意力输出dout的描述符 
-	out：输出，多头注意力输出dout的首地址 
-	weightSizeInBytes：输入，权重向量的空间大小 
-	weights：输入：权重向量的首地址 
-	dweights：输出：权重更新量向量的首地址 
-	workSpaceSizeInBytes：输入，预留空间workSpace的空间大小 
-	workSpace：输出，预留空间workspace的首地址 
-	reserveSpaceSizeInBytes：输入，预留空间reserveSpace的空间大小 
-	reserveSpace：输入/输出，预留空间reserveSpace的首地址

**支持模式attnMode_t：**

1. SWDNN_ATTN_QUERYMAP_ALL_TO_ONE

2. SWDNN_ATTN_QUERYMAP_ONE_TO_ONE

3. SWDNN_ATTN_DISABLE_PROJ_BIASES

**权重更新模式：**

1. SWDNN_WGRAD_MODE_ADD

2. SWDNN_WGRAD_MODE_SET

**目前已经支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2. SWDNN_DATA_HALF

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

## swdnnMultiHeadAttnBackwardData

```C
swdnnMultiHeadAttnBackwardData(swdnnHandle_t handle, 
	const swdnnAttnDescriptor_t attnDesc, 
	const int loWinIdx[], 
	const int hiWinIdx[], 
	const int devSeqLengthsDQDO[], 
	const int devSeqLengthsDKDV[], 
	const swdnnSeqDataDescriptor_t doDesc, 
	const void *dout, 
	const swdnnSeqDataDescriptor_t dqDesc, 
	void *dqueries, 
	const void *queries, 
	const swdnnSeqDataDescriptor_t dkDesc, 
	void *dkeys, 
	const void *keys, 
	const swdnnSeqDataDescriptor_t dvDesc, 
	void *dvalues, 
	const void *values, 
	size_t weightSizeInBytes, 
	const void *weights, 
	size_t workSpaceSizeInBytes, 
	void *workSpace, 
	size_t reserveSpaceSizeInBytes, 
	void *reserveSpace)
```

**功能描述**：

计算多头注意力函数反向过程的梯度值。

**参数描述：**

-	handle：输入，句柄 
-	attnDesc：输入，attn描述符 
-	loWinIdx\[\]：输入，描述每个Q时间步对应的attention窗口起始位置的矩阵 
-	hiWinIdx\[\]：输入，描述每个Q时间步对应的attention窗口结束位置的矩阵 
-	devSeqLengthsDQDO\[\]：输入，设备端存储dq或do的序列长度的矩阵 
-	devSeqLengthsDKDV\[\]：输入，设备端存储dk或dv的序列长度的矩阵 
-	doDesc：输入，梯度输出向量dout的描述符 
-	dout：输入，梯度输出向量dout的首地址 
-	dqDesc：输入，查询输入梯度dqueries的描述符 
-	dqueries：输出，查询输入梯度dqueries的首地址 
-	queries：输入，查询输入queries的首地址 
-	dkDesc：输入，键输入梯度dkeys的描述符 
-	dkeys：输出，键输入梯度dkeys的首地址 
-	keys：输入，键输入keys的首地址 
-	dvDesc：输入，值输入梯度dvalues的描述符 
-	dvalues：输出，值输入梯度dvalues的首地址 
-	values：输入，值输入values的首地址 
-	weightSizeInBytes：输入，权重向量的空间大小 
-	weights：输入：权重向量的首地址 
-	workSpaceSizeInBytes：输入，workSpace的空间大小 
-	workSpace：输出，workspace的首地址 
-	reserveSpaceSizeInBytes：输入，reserveSpace的空间大小 
-	reserveSpace：输入/输出，reserveSpace的首地址

**目前支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**支持模式attnMode_t：**

1. SWDNN_ATTN_QUERYMAP_ALL_TO_ONE

2. SWDNN_ATTN_QUERYMAP_ONE_TO_ONE

**权重更新模式：**

1. SWDNN_WGRAD_MODE_ADD

2. SWDNN_WGRAD_MODE_SET

## swdnnMultiHeadAttnForward

```C
swdnnMultiHeadAttnForward(swdnnHandle_t handle, 
	const swdnnAttnDescriptor_t attnDesc, 
	int currIdx, 
	const int loWinIdx[], 
	const int hiWinIdx[], 
	const int devSeqLengthsQO[], 
	const int devSeqLengthsKV[], 
	const swdnnSeqDataDescriptor_t qDesc, 
	const void *queries, 
	const void *residuals, 
	const swdnnSeqDataDescriptor_t kDesc, 
	const void *keys, 
	const swdnnSeqDataDescriptor_t vDesc, 
	const void *values, 
	const swdnnSeqDataDescriptor_t oDesc, 
	void *out, 
	size_t weightSizeInBytes, 
	const void *weights, 
	size_t workSpaceSizeInBytes, 
	void *workSpace, 
	size_t reserveSpaceSizeInBytes, 
	void *reserveSpace)
```

**功能描述**：

计算多头attention正向过程。当reserveSpaceSizeInBytes=0并且reserveSpace=NULL时，执行推理模式，否则，执行训练模式。

**参数描述：**

-	handle：输入，控制句柄 
-	attnDesc：输入，attn描述符 
-	currIdx：输入，要处理的queries时间步。当currIdx为负数时，要计算所有的queries时间步；当currIdx为0或正数时，正向过程只计算选择的时间步。后者只能用于推理模式 
-	loWinIdx\[\]：输入，描述每个Q时间步对应的attention窗口起始位置的矩阵 
-	hiWinIdx\[\]：输入，描述每个Q时间步对应的attention窗口结束位置的矩阵 
-	devSeqLengthsQO\[\]：输入，设备端存储q或o的序列长度的矩阵 
-	devSeqLengthsKV\[\]：输入，设备端存储k或v的序列长度的矩阵 
-	qDesc：输入，查询输入queries和残余输入residuals的描述符 
-	queries：输入，查询输入queries的首地址 
-	residuals：输入，残余输入residuals的首地址。没有残余连接时，该参数设置为NULL 
-	kDesc：输入，键输入keys的描述符 
-	keys：输入，键输入keys的首地址 
-	vDesc：输入，值输入values的描述符 
-	values：输入，值输入values的首地址 
-	oDesc：输入，多头attention输出out的描述符 
-	out：输出，多头attention输出out的首地址 
-	weightSizeInBytes：输入，权重向量的空间大小 
-	weights：输入：权重向量的首地址 
-	workSpaceSizeInBytes：输入，workSpace的空间大小 
-	workSpace：输出，workspace的首地址 
-	reserveSpaceSizeInBytes：输入，eserveSpace的空间大小。推理模式时为零，训练时非零 
-	reserveSpace：输入/输出，reserveSpace的首地址。推理模式时为空，训练模式时非空

**目前支持的数据类型swdnnDataType_t：**

3.  SWDNN_DATA_FLOAT

4.  SWDNN_DATA_HALF

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**支持模式attnMode_t：**

1. SWDNN_ATTN_QUERYMAP_ALL_TO_ONE

2. SWDNN_ATTN_QUERYMAP_ONE_TO_ONE

3. SWDNN_ATTN_DISABLE_PROJ_BIASES（暂不支持）

**权重更新模式：**

1. SWDNN_WGRAD_MODE_ADD

2. SWDNN_WGRAD_MODE_SET

## swdnnNegTensor

```C
swdnnNegTensor(
    swdnnHandle_t handle, 
    const swdnnTensorDescriptor_t aDesc,
    const void *A, 
    const swdnnTensorDescriptor_t cDesc, 
    void *C);
```
**功能描述**：

张量逐元素翻转y = -x运算

**参数描述**：

-	handle：输入，控制句柄 
-	alpha, beta：输入，扩展因子 
-	aDesc：输入，A的张量描述符 
-	A：输入，A的首地址指针 
-	cDesc：输入，C的张量描述符 
-	C：输入/输出，C的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC

## swdnnNllLossBackward

```C
swdnnStatus_t swdnnNllLossBackward( 
	swdnnHandle_t swdnnHandle, 
	int ignore_index, 
	unsigned reduction, 
	swdnnTensorDescriptor_t targetDesc, 
	const void* target, 
	swdnnTensorDescriptor_t weightDesc, 
	const void* weight, 
	swdnnTensorDescriptor_t doutDesc, 
	const void* dout, 
	const void* total_weight, 
	swdnnTensorDescriptor_t dinputDesc, 
	void* dinput)
```

**功能描述**：

Nll_Loss的反向计算过程

**参数描述**

-   swdnnHandle：输入，句柄
-   ignore_index：输入，当target中的值为ignore_index时，不更新梯度值
-   reduction：输入，”None”模式，或者”Mean”模式（默认模式）
-   targetDesc：输入，标签target描述符
-   target：输入，target首地址
-   weightDesc：输入，权重weight描述符
-   weight：输入，weight首地址，默认模式下可以为空指针
-   doutDesc：输入，梯度dout描述符
-   dout：输入，dout首地址；
-   total_weight：输入，当target不为ignore_index时，对应的weight值的累加和
-   dinputDesc：输入，dinput描述符
-   dinput：输出，dinput首地址

**返回值**：

1.  SWDNN_STATUS_SUCCESS成功

**备注：**

1. 支持float和half计算

## swdnnNllLossForward

```C
swdnnStatus_t swdnnNllLossForward( 
	swdnnHandle_t swdnnHandle, 
	int ignore_index, 
	unsigned reduction, 
	swdnnSeqTensor_t inputDesc, 
	const void* input, 
	swdnnTensorDescriptor_t targetDesc, 
	const void* target, 
	swdnnTensorDescriptor_t wDesc, 
	const void* weight, 
	swdnnTensorDescriptor_t outDesc, 
	void* out, 
	void* total_weight)
```

**功能描述**：

NllLoss的前向计算过程

**参数描述**：

-   swdnnHandle：输入，句柄
-   ignore_index：输入，当target中的值为ignore_index时，loss为0，对应的weight中的值不参与累加
-   reduction：输入，描述NllLoss计算模式
-   inputDesc：输入，input描述符
-   input：输入，input首地址
-   targetDesc：输入，标签target描述符
-   target：输入，target首地址
-   weightDesc：输入，权重weight描述符
-   weight：输入，weight首地址，为空指针时，即默认模式，值均为1
-   outDesc：输入，out描述符
-   out：输出，out首地址
-   total_weight：输出，当target不为ignore_index时，对应的weight值的累加和

**返回值**：

1.  SWDNN_STATUS_SUCCESS成功

**备注：**

1. 支持float和half计算

## swdnnOpTensor

```C
swdnnOpTensor(swdnnHandle_t handle, 
	const swdnnOpTensorDescriptor_t opTensorDesc, 
	const void *alpha1, 
	const swdnnTensorDescriptor_t aDesc, 
	const void *A, 
	const void *alpha2, 
	const swdnnTensorDescriptor_t bDesc, 
	const void *B, 
	const void *beta, 
	const swdnnTensorDescriptor_t cDesc, 
	void *C);
```

**功能描述**：

进行OpTensor计算（C = op( alpha1 * A, alpha2 * B ) + beta * C）

**参数描述**：

-   handle：输入，设备句柄
-   opTensorDesc：输入，OpTensor描述符
-   alpha1：输入
-   aDesc：输入，输入数据A描述符
-   A：输入
-   alpha2：输入
-   bDesc：输入，输入数据B描述符
-   B：输入
-   beta：输入
-   cDesc：输入，输出数据C描述符

C：输出

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

**note**：

1. 支持float和hlaf数据


## swdnnPoolingBackward

```C
swdnnStatus_t swdnnPoolingBackward( 
	swdnnHandle_t handle, 
	const swdnnPoolingDescriptor_t poolingDesc, 
	const void *alpha, 
	const swdnnTensorDescriptor_t yDesc, 
	const void *y, 
	const swdnnTensorDescriptor_t dyDesc, 
	const void *dy, 
	const swdnnTensorDescriptor_t xDesc, 
	const void *xData, 
	const void *beta, 
	const swdnnTensorDescriptor_t dxDesc, 
	void *dx)
```

**功能描述**：

反向过程中池化函数的梯度值计算

**参数描述**：

-   handle：输入，控制句柄
-   poolingDesc：输入，池化描述符
-   alpha, beta：输入，扩展因子，目前未用
-   yDesc：输入，y的张量描述符
-   y：输入，y的首地址指针
-   dyDesc：输入，dy的张量描述符
-   dy：输入，y的首地址指针
-   xDesc：输入，x的张量描述符
-   x：输入，x的首地址指针
-   dxDesc：输入，dx的张量描述符
-   dx：输出，dx的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1. SWDNN_TENSOR_NHWC

**目前已支持的池化模式：swdnnPoolingMode_t：**

1. SWDNN_POOLING_MAX

2. SWDNN_POOLING_MAX_DETERMINISTIC

3. SWDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING

4. SWDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING

## swdnnPoolingForward

```C
swdnnStatus_t swdnnPoolingForward( 
	swdnnHandle_t handle, 
	const swdnnPoolingDescriptor_t poolingDesc, 
	const void *alpha, 
	const swdnnTensorDescriptor_t xDesc, 
	const void *x, 
	const void *beta, 
	const swdnnTensorDescriptor_t yDesc, 
	void *y)
```

**功能描述**：

正向过程中池化函数的池化值计算

**参数描述**：

-   handle：输入，控制句柄
-   poolingDesc：输入，池化描述符
-   alpha, beta：输入，扩展因子，目前未用
-   xDesc：输入，x的张量描述符
-   x：输入，x的首地址指针
-   yDesc：输入，y的张量描述符
-   y：输出，y的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1. SWDNN_DATA_FLOAT

2. SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1. SWDNN_TENSOR_NHWC

**目前已支持的池化模式：swdnnPoolingMode_t：**

1. SWDNN_POOLING_MAX

2. SWDNN_POOLING_MAX_DETERMINISTIC

3. SWDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING

4. SWDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING

## swdnnRandomUniform

**（暂不支持）**

```C
swdnnStatus_t SWDNNWINAPI
swdnnRandomUniform(
	swdnnHandle_t handle, 
	unsigned long long seed, 
	float min,
    float max, 
	const swdnnTensorDescriptor_t dataDesc, 
	void *data);
```

**功能描述**：

在满足最大最小限制的条件下，以均匀分布的方式填充张量。

**参数描述**：

-   handle：输入，控制句柄
-   seed: 输入，种子
-   min: 输入，最小值
-   max: 输入，最大值
-   dataDesc：输入，数据张量描述符
-   data：输出，输出张量指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

## swdnnReduceTensor

```C
swdnnReduceTensor(swdnnHandle_t handle, 
	const swdnnReduceTensorDescriptor_t reduceTensorDesc, 
	void *indices, 
	size_t indicesSizeInBytes, 
	void *workspace, 
	size_t workspaceSizeInBytes, 
	const void *alpha, 
	const swdnnTensorDescriptor_t aDesc, 
	const void *A, 
	const void *beta, 
	const swdnnTensorDescriptor_t cDesc, 
	void *C);
```

**功能描述**：

进行reduce计算

**参数描述**：

-   handle：输入，设备句柄
-   reduceTensorDesc：输入，reduce操作描述符
-   indices: 输出，结果的索引值
-   indicesSizeInBytes：输入，索引空间的大小
-   workspace：输出，计算的工作区
-   workspaceSizeInBytes：输入，计算的工作区大小
-   alpha：输入
-   aDesc：输入，需要被reduce的数据描述符
-   A：输入，需要reduce的数据
-   beta：输入
-   cDesc：输入，reduce的结果数据描述符
-   C：输出，reduce后的结果

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

**note**：

1. 支持float和half计算

2. 对于某些reduce操作，如果reduce数据量过大，会导致精度降低

## swdnnRestoreDropoutDescriptor

**（暂不支持）**

```C
swdnnStatus_t swdnnRestoreDropoutDescriptor( 
	swdnnDropoutDescriptor_t dropoutDesc, 
	swdnnHandle_t handle, 
	float dropout, 
	void *states, 
	size_t stateSizeInBytes, 
	unsigned long long seed)
```

**功能描述**：

恢复dropout描述符

**参数描述**：

-   dropoutDesc：输入/输出，dropout描述符
-   handle：输入，控制句柄
-   dropout：输入，进行dropout的概率值
-   states：输出，随机数生成状态值地址
-   stateSizeInBytes：输入，随机数生成状态空间大小
-   seed：输入，随机数生成种子

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

## swdnnRNNBackwardData

```C
swdnnStatus_t swdnnRNNBackwardData( 
	swdnnHandle_t handle, 
	const swdnnRNNDescriptor_t rnnDesc, 
	const int seqLength, 
	const swdnnTensorDescriptor_t *yDesc, 
	const void *y, 
	const swdnnTensorDescriptor_t *dyDesc, 
	const void *dy, 
	const swdnnTensorDescriptor_t dhyDesc, 
	const void *dhy, 
	const swdnnTensorDescriptor_t dcyDesc, 
	const void *dcy, 
	const swdnnFilterDescriptor_t wDesc, 
	const void *w, 
	const swdnnTensorDescriptor_t hxDesc, 
	const void *hx, 
	const swdnnTensorDescriptor_t cxDesc, 
	const void *cx, 
	const swdnnTensorDescriptor_t *dxDesc, 
	void *dx, 
	const swdnnTensorDescriptor_t dhxDesc, 
	void *dhx, 
	const swdnnTensorDescriptor_t dcxDesc, 
	void *dcx, 
	void *workspace, 
	size_t workSpaceSizeInBytes, 
	const void *reserveSpace, 
	size_t reserveSpaceSizeInBytes)
```

**功能描述**：

计算RNN反向过程的梯度。

**参数描述：**

-   handle：输入，句柄
-   rnnDesc：输入，rnn描述符
-   seqLength：输入，时间序列长度
-   yDesc：输入，y的描述符
-   y：输入，y的地址
-   dyDesc：输入，dy的描述符
-   dy：输入，dy的地址
-   dhyDesc：输入，向量dhy的描述符
-   dhy：输入，向量dhy的地址
-   dcyDesc：输入，向量dcy的描述符
-   dcy：输入，向量dcy的地址
-   wDesc：输入，w的描述符
-   w：输入，w的地址
-   hxDesc：输入，向量hx的描述符
-   hx：输入，向量hx的地址
-   cxDesc：输入，向量cx的描述符
-   cx：输入，向量cx的地址
-   dxDesc：输入，向量dx的描述符
-   dx：输出，向量dx的地址
-   dhxDesc：输入，向量dhx的描述符
-   dhx：输出，向量dhx的地址
-   dcxDesc：输入，向量dcx的描述符
-   dcx：输出，向量dcx的地址
-   workspace：输出，workspace的地址
-   workSpaceSizeInBytes：输入，workspace的空间大小
-   reserveSpace：输入/输出，reserveSpace的地址
-   reserveSpaceSizeInBytes：输入，reserveSpace的空间大小

**支持模式swdnnRNNMode_t：**

1. SWDNN_RNN_RELU

2. SWDNN_RNN_TANH

3. SWDNN_LSTM

4. SWDNN_GRU

**支持的输入模式swdnnRNNInputMode_t：**

1. SWDNN_LINEAR_INPUT

2. SWDNN_SKIP_INPUT

**支持的循环方向swdnnDirectionMode_t：**

1. SWDNN_UNIDIRECTIONAL

2. SWDNN_BIDIRECTIONAL

**支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**返回值：**

1. WDNN_STATUS_SUCCESS成功

2. WDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. WDNN_STATUS_ALLOC_FAILED分配内存失败

4. WDNN_STATUS_BAD_PARAM参数错误

## swdnnRNNBackwardWeights

```C
swdnnStatus_t swdnnRNNBackwardWeights( 
	swdnnHandle_t handle, 
	const swdnnRNNDescriptor_t rnnDesc, 
	const int seqLength, 
	const swdnnTensorDescriptor_t *xDesc, 
	const void *x, 
	const swdnnTensorDescriptor_t hxDesc, 
	const void *hx, 
	const swdnnTensorDescriptor_t *yDesc, 
	const void *y, 
	const void *workspace, 
	size_t workSpaceSizeInBytes, 
	const swdnnFilterDescriptor_t dwDesc, 
	void *dw, 
	const void *reserveSpace, 
	size_t reserveSpaceSizeInBytes)
```

**功能描述**：

RNN反向过程中权值参数更新

**参数描述：**

-   handle：输入，控制句柄
-   rnnDesc：输入，RNN描述符
-   seqLength：输入，时序长度
-   xDesc：输入，x的张量描述符
-   x：输入，x的首地址指针
-   hxDesc：输入，hx的张量描述符
-   hx：输入，hx的首地址指针
-   yDesc：输入，y的张量描述符
-   y：输入，y的首地址指针
-   dwDesc：输入，dw的张量描述符
-   dw：输出，dw的首地址指针
-   reserveSpace：输入
-   reserveSpaceSizeInBytes：输入，reserveSpace的空间大小

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前RNN支持的模式swdnnRNNMode_t：**

1. SWDNN_RNN_RELU

2. SWDNN_RNN_TANH

3. SWDNN_LSTM

4. SWDNN_GRU

**RNN支持的第一层模式swdnnRNNInputMode_t：**

1. SWDNN_LINEAR_INPUT

2. SWDNN_SKIP_INPUT

**支持的循环方向swdnnDirectionMode_t：**

1. SWDNN_UNIDIRECTIONAL

2. SWDNN_BIDIRECTIONAL

## swdnnRNNForwardInference

```C
swdnnStatus_t swdnnRNNForwardInference( 
	swdnnHandle_t handle, 
	const swdnnRNNDescriptor_t rnnDesc, 
	const int seqLength, 
	const swdnnTensorDescriptor_t *xDesc, 
	const void *x, 
	const swdnnTensorDescriptor_t hxDesc, 
	const void *hx, 
	const swdnnTensorDescriptor_t cxDesc, 
	const void *cx, 
	const swdnnFilterDescriptor_t wDesc, 
	const void *w, 
	const swdnnTensorDescriptor_t *yDesc, 
	void *y, 
	const swdnnTensorDescriptor_t hyDesc, 
	void *hy, 
	const swdnnTensorDescriptor_t cyDesc, 
	void *cy, 
	void *workspace, 
	size_t workSpaceSizeInBytes)
```

**功能描述**：

RNN前向推理

**参数描述：**

-   handle：输入，句柄
-   rnnDesc：输入，rnn描述符
-   seqLength：输入，时间序列长度
-   xDesc：输入，x的描述符
-   x：输出，梯度x的地址
-   hxDesc：输入，隐藏层hx的描述符
-   hx：输入，隐藏层hx的地址
-   cxDesc：输入，cx的描述符
-   cx：输入，cx的地址
-   wDesc：输入，w的描述符
-   w：输入，w的地址
-   yDesc：输入，y的描述符
-   y：输出，y的地址
-   hyDesc：输入，隐藏层hy的描述符
-   hy：输出，隐藏层hy的地址
-   cyDesc：输入，cy的描述符
-   cy：输出，cy的地址
-   workSpaceSizeInBytes：输入，workspace的空间大小
-   workSpace：输出，workSpace的地址

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前支持的数据类型swdnnDataType_t：**

1. SWDNN_DATA_FLOAT

**目前支持的模式swdnnRNNMode_t：**

1. SWDNN_RNN_RELU

2. SWDNN_RNN_TANH

3. SWDNN_LSTM

4. SWDNN_GRU

**RNN第一层模式swdnnRNNInputMode_t：**

1. SWDNN_LINEAR_INPUT

2. SWDNN_SKIP_INPUT

**支持的循环方向swdnnDirectionMode_t：**

1. SWDNN_UNIDIRECTIONAL

2. SWDNN_BIDIRECTIONAL

## swdnnRNNForwardTraining

```C
swdnnStatus_t swdnnRNNForwardTraining( 
	swdnnHandle_t handle, 
	const swdnnRNNDescriptor_t rnnDesc, 
	const int seqLength, 
	const swdnnTensorDescriptor_t *xDesc, 
	const void *x, 
	const swdnnTensorDescriptor_t hxDesc, 
	const void *hx, 
	const swdnnTensorDescriptor_t cxDesc, 
	const void *cx, 
	const swdnnFilterDescriptor_t wDesc, 
	const void *w, 
	const swdnnTensorDescriptor_t *yDesc, 
	void *y, 
	const swdnnTensorDescriptor_t hyDesc, 
	void *hy, 
	const swdnnTensorDescriptor_t cyDesc, 
	void *cy, 
	void *workspace, 
	size_t workSpaceSizeInBytes, 
	void *reserveSpace, 
	size_t reserveSpaceSizeInBytes)
```

**功能描述**：

RNN前向训练

**参数描述：**

-   handle：输入，句柄
-   rnnDesc：输入，rnn描述符
-   seqLength：输入，时间序列长度
-   xDesc：输入，x的描述符
-   x：输出，梯度x的地址
-   hxDesc：输入，隐藏层hx的描述符
-   hx：输入，隐藏层hx的地址
-   cxDesc：输入，cx的描述符
-   cx：输入，cx的地址
-   wDesc：输入，w的描述符
-   w：输入，w的地址
-   yDesc：输入，y的描述符
-   y：输出，y的地址
-   hyDesc：输入，隐藏层hy的描述符
-   hy：输出，隐藏层hy的地址
-   cyDesc：输入，cy的描述符
-   cy：输出，cy的地址
-   workSpaceSizeInBytes：输入，workspace的空间大小
-   workspace：输出，workSpace地址
-   reserveSpace：输入/输出，reserveSpace的地址
-   reserveSpaceSizeInBytes：输入，reserveSpace的空间大小

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**支持的数据类型swdnnDataType_t：**

1. SWDNN_DATA_FLOAT

**支持的模式swdnnRNNMode_t：**

1. SWDNN_RNN_RELU

2. SWDNN_RNN_TANH

3. SWDNN_LSTM

4. SWDNN_GRU

**RNN输入模式swdnnRNNInputMode_t：**

1. SWDNN_LINEAR_INPUT

2. SWDNN_SKIP_INPUT

**支持的循环方向swdnnDirectionMode_t：**

1. SWDNN_UNIDIRECTIONAL

2. SWDNN_BIDIRECTIONAL

## swdnnRoiAlignBackward

```C
swdnnRoiAlignBackward(
    swdnnHandle_t handle,
    int aligned, 
    float scale, 
    int ratio,
    const swdnnTensorDescriptor_t boxesDesc, 
    void *boxes,
    const swdnnTensorDescriptor_t dyDesc, 
    const void *dy,
    const swdnnTensorDescriptor_t dxDesc, 
    const void *dx);
```

**功能描述**：

使用反向平均池执行感兴趣区域 (RoI) 对齐算子

**参数描述**：

-   handle：输入，控制句柄
-   aligned: 输入，设置坐标偏移，
-   scale: 输入，将框坐标映射到输入坐标的比例因子
-   ratio: 输入，插值网格中的采样点数
-   boxes: 输入，从中获取区域的框坐标。
-   boxesDesc：输入，boxes的张量描述符
-   dx：输出，dx的首地址指针
-   dxDesc: 输入，dx的张量描述符
-   dy：输入，dy的首地址指针
-   dyDesc：输入，dy的张量描述符

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1. SWDNN_DATA_FLOAT

2. SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1. SWDNN_TENSOR_NHWC

## swdnnRoiAlignForward

```C
swdnnRoiAlignForward(
    swdnnHandle_t handle,
    int aligned, 
    float scale, 
    int ratio,
    const swdnnTensorDescriptor_t boxesDesc, 
    void *boxes,
    const swdnnTensorDescriptor_t xDesc, 
    const void *x,  
    const swdnnTensorDescriptor_t yDesc, 
    const void *y);
```

**功能描述**：

使用正向平均池执行感兴趣区域 (RoI) 对齐算子

**参数描述**：

-   handle：输入，控制句柄
-   aligned: 输入，设置坐标偏移，
-   scale: 输入，将框坐标映射到输入坐标的比例因子
-   ratio: 输入，插值网格中的采样点数
-   boxes: 输入，从中获取区域的框坐标。
-   boxesDesc：输入，boxes的张量描述符
-   x：输入，x的首地址指针
-   xDesc: 输入，x的张量描述符
-   y：输出，y的首地址指针
-   yDesc：输入，y的张量描述符

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1. SWDNN_DATA_FLOAT

2. SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1. SWDNN_TENSOR_NHWC

## swdnnRoiPoolBackward
```C
swdnnRoiPoolBackward(
    swdnnHandle_t handle,
    float spatial,
    const swdnnTensorDescriptor_t boxesDesc, 
    void *boxes,
    const swdnnTensorDescriptor_t dyDesc, 
    const void *dy,
    const swdnnTensorDescriptor_t dxDesc, 
    const void *dx);
```
**功能描述**：

执行反向感兴趣区域 (RoI) 池算子

**参数描述**：

-   handle：输入，控制句柄
-   spatial：输入，将框坐标映射到输入坐标的比例因子
-   boxes: 输入，从中获取区域的框坐标。
-   boxesDesc：输入，boxes的张量描述符
-   dx：输出，dx的首地址指针
-   dxDesc: 输入，dx的张量描述符
-   dy：输入，dy的首地址指针
-   dyDesc：输入，dy的张量描述符

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1. SWDNN_DATA_FLOAT

2. SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1. SWDNN_TENSOR_NHWC

## swdnnRoiPoolForward

```C
swdnnRoiPoolForward(
    swdnnHandle_t handle,
    float spatial,
    const swdnnTensorDescriptor_t boxesDesc, 
    void *boxes,
    const swdnnTensorDescriptor_t xDesc, 
    const void *x,  
    const swdnnTensorDescriptor_t yDesc, 
    const void *y);
```
**功能描述**：

执行正向感兴趣区域 (RoI) 池算子

**参数描述**：

-   handle：输入，控制句柄
-   spatial：输入，将框坐标映射到输入坐标的比例因子
-   boxes: 输入，从中获取区域的框坐标。
-   boxesDesc：输入，boxes的张量描述符
-   x：输入，x的首地址指针
-   xDesc: 输入，x的张量描述符
-   y：输出，y的首地址指针
-   yDesc：输入，y的张量描述符

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1. SWDNN_DATA_FLOAT

2. SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1. SWDNN_TENSOR_NHWC

## swdnnScaleTensor

```C
swdnnStatus_t swdnnScaleTensor( 
	swdnnHandle_t handle, 
	const swdnnTensorDescriptor_t yDesc, 
	void *y, 
	const void *alpha)
```

**功能描述**：

将所有张量按给定值进行缩放

**参数描述**：

-   handle：输入，控制句柄
-   yDesc：输入，y张量描述符
-   y：输入\输出，y张量值地址
-   alpha：输入，给定值地址

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**备注：**

1. 支持float和half计算

2. 仅支持数据任意4维格式存储

## swdnnSetActivationDescriptor

```C
swdnnStatus_t swdnnSetActivationDescriptor( 
	swdnnActivationDescriptor_t activationDesc, 
	swdnnActivationMode_t mode, 
	swdnnNanPropagation_t reluNanOpt, 
	float coef)
```

**功能描述**：

设置激活描述符的参数值

**参数描述**：

-   activationDesc：输入/输出，激活描述符
-   mode，输入，激活模式
-   reluNanOpt：输入，Nan传播模式
-   coef：输入，系数，在CRELU中代表C值，在ELU中代表alpha值。

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnSetAttnDescriptor

```C
swdnnSetAttnDescriptor(swdnnAttnDescriptor_t attnDesc, 
	unsigned attnMode, 
	int nHeads, 
	double smScaler, 
	swdnnDataType_t dataType, 
	swdnnDataType_t computePrec, 
	swdnnMathType_t mathType, 
	swdnnDropoutDescriptor_t attnDropoutDesc, 
	swdnnDropoutDescriptor_t postDropoutDesc, 
	int qSize, 
	int kSize, 
	int vSize, 
	int qProjSize, 
	int kProjSize, 
	int vProjSize, 
	int oProjSize, 
	int qoMaxSeqLength, 
	int kvMaxSeqLength, 
	int maxBatchSize, 
	int maxBeamSize);
```

**功能描述**：

设置神经元函数参数值

**参数描述：**

-   attnMode：神经元函数模式
-   nHeads：输入
-   smScaler：输入，缩放系数dataType：输入，attn数据类型
-   computePrec：输入，计算数据类型
-   mathType：输入，计算模式
-   attnDropoutDesc：输入，attnDropout描述符（暂不支持）
-   postDropoutDesc：输入，postDropout描述符（暂不支持）
-   qSize：输入，qSize值
-   kSize：输入，kSize值
-   vSize：输入，kSize值
-   qProjSize：输入，qProjSize值
-   kProjSize：输入，kProjSize值
-   vProjSize：输入，vProjSize值
-   oProjSize：输入，oProjSize值
-   qoMaxSeqLength：输入，查询序列的最大长度
-   kvMaxSeqLength：输入，隐藏层序列的最大长度
-   maxBatchSize：输入，最大batch值
-   maxBeamSize：输入，最大Beam值

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnSetConvolution2dDescriptor

```C
swdnnStatus_t swdnnSetConvolution2dDescriptor( 
	swdnnConvolutionDescriptor_t convDesc, 
	int pad_h, 
	int pad_w, 
	int u, 
	int v, 
	int dilation_h, 
	int dilation_w, 
	swdnnConvolutionMode_t mode, 
	swdnnDataType_t computeType)
```

**功能描述**：

设置二维卷积描述符的参数值

**参数描述**：

-   convDesc：输入\输出，卷积描述符
-   pad_h：输入，h方向的padding值
-   pad_w：输入，w方向的padding值
-   u：输入，垂直方向的卷积核跨步
-   v：输入，水平方向的卷积核跨步
-   dilation_h：输入，h方向的dilation值
-   dilation_w：输入，w方向的dilation值
-   mode：输入，卷积模式
-   DataType：输入，卷积数据类型

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnSetConvolutionGroupCount

```C
swdnnStatus_t swdnnSetConvolutionGroupCount(swdnnConvolutionDescriptor_t convDesc,int groupCount)
```

**功能描述**：

设置卷积的分组参数值

**参数描述**：

-   convDesc：输入，卷积描述符
-   groupCount：输入，分组值

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnSetConvolutionMathType

```C
swdnnStatus_t swdnnSetConvolutionMathType( 
	swdnnConvolutionDescriptor_t convDesc, 
	swdnnMathType_t mathType)
```

**功能描述**：

设置卷积的数学类型

**参数描述：**

-   convDesc：输入\输出，卷积描述符

-   mathType：输入，卷积数学类型

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

## swdnnSetConvolutionNdDescriptor

```C
swdnnStatus_t swdnnSetConvolutionNdDescriptor( 
	swdnnConvolutionDescriptor_t convDesc, 
	int arrayLength, 
	const int padA[], 
	const int filterStrideA[], 
	const int dilationA[], 
	swdnnConvolutionMode_t mode, 
	swdnnDataType_t dataType)
```

**功能描述**：

设置N维卷积描述符的参数值

**参数描述**：

-   convDesc：输入\输出，卷积描述符
-   arrayLength：输入，实际的卷积维度
-   padA：输入，各个方向的padding值
-   filterStrideA：输入，各个方向的卷积核跨步
-   dilationA：输入，各个方向的dilation值
-   mode：输入，卷积模式
-   DataType：输入，卷积数据类型

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnSetDropoutDescriptor

```C
swdnnStatus_t swdnnSetDropoutDescriptor( 
	swdnnDropoutDescriptor_t dropoutDesc, 
	swdnnHandle_t handle, 
	float dropout, 
	void *states, 
	size_t stateSizeInBytes, 
	unsigned long long seed)
```

**功能描述**：

设置Dropout描述符

**参数描述**：

-   dropoutDesc：输入\输出，dropout描述符
-   handle：输入
-   dropout：输入，丢弃因子(\[0,1\])
-   states：输出，states的首地址指针
-   stateSizeInBytes：输入，state空间大小
-   seed：输入，随机种子

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnSetEmbeddingDescriptor

```C
swdnnStatus_t swdnnSetEmbeddingDesciptor( 
	swdnnEmbeddingDesciptor_t embeddingDesc, 
	int64_t padding_idx, 
	unsigned scale_grad_mode, 
	unsigned array_type)
```

**功能描述**：

设置Embedding描述符

**参数描述：**

-   embeddingDesc：输入/输出，embedding描述符
-   padding_idx：输入，padding方式
-   scale_grad_mode：输入，梯度计算模式
-   array_type ：输入，矩阵类型

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

## swdnnSetFilter4dDescriptor

```C
swdnnStatus_t swdnnSetFilter4dDescriptor( 
	swdnnFilterDescriptor_t filterDesc, 
	swdnnDataType_t dataType, 
	swdnnTensorFormat_t format, 
	int k, 
	int c, 
	int h, 
	int w)
```

**功能描述**：

设置四维滤波描述符

**参数描述**：

-   filterDesc：输入/输出，滤波描述符
-   dataType：输入，滤波的数据类型
-   format：输入，滤波的数据格式
-   k：输入，滤波的个数值
-   c：输入，滤波的通道数值
-   h：输入，滤波的高度值
-   w：输入，滤波的宽度值

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnSetFilterNdDescriptor

```C
swdnnStatus_t swdnnSetFilterNdDescriptor( 
	swdnnFilterDescriptor_t filterDesc, 
	swdnnDataType_t dataType, 
	swdnnTensorFormat_t format, 
	int nbDims, 
	const int filterDimA[])
```

**功能描述**：

设置N维滤波参数值

**参数描述**：

-   tensorDesc：输入\输出，滤波描述符
-   dataType：输入，滤波的数据类型
-   format：输入，滤波的数据格式
-   nbDims：输入，实际的张量维度
-   filterDimA：输入，各个方向滤波的维度值

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnSetFusedOpsConstParamPackAttribute

```C
swdnnSetFusedOpsConstParamPackAttribute(swdnnFusedOpsConstParamPack_t
	constPack, 
	swdnnFusedOpsConstParamLabel_t paramLabel, 
	const void *param)
```

**功能描述**：

设置FusedOpConstParam描述符

**参数描述**：

-   constPack：输出，设置的FusedOpConstParam描述符
-   paramLabel：输入，设置的参数标签
-   param：输入，设置的参数

**返回值**：

1. SWDNN_STATUS_SUCCESS成功


## swdnnSetFusedOpsVariantParamPackAttribute

```C
swdnnSetFusedOpsVariantParamPackAttribute( 
	swdnnFusedOpsVariantParamPack_t varPack, 
	swdnnFusedOpsVariantParamLabel_t paramLabel, 
	void *ptr)
```

**功能描述**：

设置FusedOpsVariantParam描述符

**参数描述**：

-   varPack：输出，需要设置的FusedOpsVariantParam描述符
-   paramLabel：输入，设置的参数标签
-   ptr：输入，设置的参数

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnSetOpTensorDescriptor

```C
swdnnSetOpTensorDescriptor(swdnnOpTensorDescriptor_t opTensorDesc, 
	swdnnOpTensorOp_t opTensorOp, 
	swdnnDataType_t opTensorCompType, 
	swdnnNanPropagation_t opTensorNanOpt);
```

**功能描述**：

设置Op描述符

**参数描述**：

-   opTensorDesc：输出，需要设置的Op描述符
-   opTensorOp：输入，设置的Optensor模式
-   opTensorCompType：输入，设置的计算数据类型
-   opTensorNanOpt：输入

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnSetPooling2dDescriptor

```C
swdnnStatus_t swdnnSetPooling2dDescriptor( 
	swdnnPoolingDescriptor_t poolingDesc, 
	swdnnPoolingMode_t mode, 
	swdnnNanPropagation_t maxpoolingNanOpt, 
	int windowHeight, 
	int windowWidth, 
	int verticalPadding, 
	int horizontalPadding, 
	int verticalStride, 
	int horizontalStride)
```

**功能描述**：

设置二维池化描述符的参数值

**参数描述：**

-   poolingDesc：输入\输出，池化描述符
-   mode：输入，池化模式值
-   maxpoolingNanOpt：输入，是否传播Nan
-   windowHeight：输入，池化窗口高度值
-   windowWidth：输入，池化窗口宽度值
-   verticalPadding：输入，垂直Padding值
-   horizontalPadding：输入，水平Padding值
-   verticalStride：输入，垂直跨步值
-   horizontalStride：输入，水平跨步值

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

## swdnnSetPoolingNdDescriptor

```C
swdnnStatus_t swdnnSetPoolingNdDescriptor( 
	swdnnPoolingDescriptor_t poolingDesc, 
	const swdnnPoolingMode_t mode, 
	const swdnnNanPropagation_t maxpoolingNanOpt, 
	int nbDims, 
	const int windowDimA[], 
	const int paddingA[], 
	const int strideA[])
```

**功能描述**：

设置N维池化描述符的参数值

**参数描述**：

-   poolingDesc：输入\输出，池化描述符
-   mode：输入，池化模式值
-   maxpoolingNanOpt：输入，是否传播Nan
-   nbDims：输入，实际的维度值
-   windowDimA：输入，池化窗口各个维度值
-   paddingA：输入，池化各个维度Padding值
-   strideA：输入，池化各个维度跨步值

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnSetReduceTensorDescriptor

```C
swdnnSetReduceTensorDescriptor(swdnnReduceTensorDescriptor_t
	reduceTensorDesc, 
	swdnnReduceTensorOp_t reduceTensorOp, 
	swdnnDataType_t reduceTensorCompType, 
	swdnnNanPropagation_t reduceTensorNanOpt, 
	swdnnReduceTensorIndices_t reduceTensorIndices, 
	swdnnIndicesType_t reduceTensorIndicesType);
```

**功能描述**：

设置ReduceTensor描述符。

**参数描述**：

-   reduceTensorDesc: 输出，要被初始化的ReduceTensor描述符。
-   reduceTensorOp: 输入，ReduceTensor的计算模式
-   reduceTensorCompType: 输入，ReduceTensor计算的数据类型
-   reduceTensorNanOpt: 输入
-   reduceTensorIndices: 输入，ReduceTensor的indices操作类型
-   reduceTensorIndicesType: 输入，indices的返回类型

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

**note**：

1. 暂不支持indices操作。

## swdnnSetRNNDescriptor

```C
swdnnSetRNNDescriptor(swdnnHandle_t handle, 
	swdnnRNNDescriptor_t rnnDesc, 
	const int hiddenSize, 
	const int numLayers, 
	swdnnDropoutDescriptor_t dropoutDesc, 
	swdnnRNNInputMode_t inputMode, 
	swdnnDirectionMode_t direction, 
	swdnnRNNMode_t mode, 
	swdnnRNNAlgo_t algo, 
	swdnnDataType_t mathPrec);
```

**功能描述**：

设置RNN描述符的参数值

**参数描述：**

-   rnnDesc：输入，rnn描述符
-   hiddenSize：隐藏层维度
-   numLayers：输入，隐藏层数量
-   dropoutDesc：输入，dropout描述符
-   inputMode：输入，rnn输入模式
-   direction：输入，循环方向
-   mode：网络类型
-   algo：rnn支持的算法类型
-   mathPrec：计算的数据类型

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

## swdnnSetSeqDataDescriptor

```C
swdnnSetSeqDataDescriptor(swdnnSeqDataDescriptor_t seqDataDesc, 
	swdnnDataType_t dataType, 
	int nbDims, 
	const int dimA[], 
	const swdnnSeqDataAxis_t axes[], 
	size_t seqLengthArraySize, 
	const int seqLengthArray[], 
	void *paddingFill);
```

**功能描述**：

设置SeqData描述符参数值

**参数描述：**

-   dataType：输入，RNN数据类型
-   nbDims：输入，实际的维度值
-   dimA：输入，各个方向维度值
-   axes：输入，用于配置swdnnSetSeqDataDescriptor()中的参数dimA
-   seqLengthArraySize,输入，seqLenth的矩阵大小
-   seqLengthArray：输入，seqLenthArray
-   paddingFill：输入，paddingFill首地址

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

## swdnnSetTensor

```C
swdnnStatus_t swdnnSetTensor( 
	swdnnHandle_t handle, 
	const swdnnTensorDescriptor_t yDesc, 
	void *y, 
	const void *valuePtr)
```

**功能描述**：

设置张量为一个给定值

**参数描述**：

-   handle：输入，控制句柄
-   yDesc：输入，y张量描述符
-   y：输入\输出，y张量值地址
-   valuePtr：输入，给定值地址

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**备注：**

1. 支持float和half计算

2. 仅支持数据任意4维格式存储

## swdnnSetTensor4dDescriptor

```C
swdnnStatus_t swdnnSetTensor4dDescriptor( 
	swdnnTensorDescriptor_t tensorDesc, 
	swdnnTensorFormat_t format, 
	swdnnDataType_t dataType, 
	int n, 
	int c, 
	int h, 
	int w)
```

**功能描述**：

设置四维张量描述符的参数值

**参数描述**：

-   tensorDesc：输入\输出，张量描述符
-   format：输入，张量的数据格式
-   dataType：输入，张量的数据类型
-   n：输入，张量的个数值
-   c：输入，张量的通道数值
-   h：输入，张量的高度值
-   w：输入，张量的宽度值

**返回值：**

1. SWDNN_STATUS_SUCCESS成功

## swdnnSetTensor4dDescriptorEx

```C
swdnnStatus_t swdnnSetTensor4dDescriptorEx( 
	swdnnTensorDescriptor_t tensorDesc, 
	swdnnDataType_t dataType, 
	int n, 
	int c, 
	int h, 
	int w, 
	int nStride, 
	int cStride, 
	int hStride, 
	int wStride)
```

**功能描述**：

设置四维张量描述符的参数值

**参数描述**：

-   tensorDesc：输入\输出，张量描述符
-   dataType：输入，张量的数据类型
-   n：输入，张量的个数值
-   c：输入，张量的通道数值
-   h：输入，张量的高度值
-   w：输入，张量的宽度值
-   nStride：输入，张量个数的跨步值
-   cStride：输入，张量通道数的跨步值
-   hStride：输入，张量高度的跨步值
-   wStride：输入，张量宽度的跨步值

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnSetTensorNdDescriptor

```C
swdnnStatus_t swdnnSetTensorNdDescriptor( 
	swdnnTensorDescriptor_t tensorDesc, 
	swdnnDataType_t dataType, 
	int nbDims, 
	const int dimA[], 
	const int strideA[])
```

**功能描述**：

设置N维张量描述符的参数值

**参数描述**：

-   tensorDesc：输入/输出，张量描述符
-   dataType：输入，张量的数据类型
-   nbDims：输入，实际的张量维度
-   dimA：输入，各个方向张量的维度值
-   strideA：输入，各个方向张量的跨步值

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

## swdnnSetTensorNdDescriptorEx

```C
swdnnStatus_t swdnnSetTensorNdDescriptor( 
	swdnnTensorDescriptor_t tensorDesc, 
	swdnnTensorFormat_t format, 
	swdnnDataType_t dataType, 
	int nbDims, 
	const int dimA[])
```

**功能描述**：

设置N维张量描述符的参数值

**参数描述**：

-   tensorDesc：输入\输出，张量描述符
-   format：输入，张量的数据格式
-   dataType：输入，张量的数据类型
-   nbDims：输入，实际的张量维度
-   dimA：输入，各个方向张量的维度值

**返回值**：

1)SWDNN_STATUS_SUCCESS成功

## swdnnSetTensorTransformDescriptor

```C
swdnnStatus_t SWDNNWINAPI 
	swdnnSetTensorTransformDescriptor( 
	swdnnTensorTransformDescriptor_t transformDesc, 
	const unsigned int nbDims, 
	const swdnnTensorFormat_t destFormat, 
	const int padBeforeA[], 
	const int padAfterA[], 
	const unsigned int foldA[], 
	const swdnnFoldingDirection_t direction)
```

**功能描述**：

设置TensorTransform描述符。

**参数描述：**

-   transformDesc：输出，要被初始化的TensorTransform描述符。
-   nbDims：输入,，指示padBeforeA、padAfterA、foldA三个数组的长度（暂时支持为4. 
-   destFormat：输入，目标张量的格式，目前只支持NHWC
-   padBeforeA[]：输入，padBefore数组，padAfter数组，长度等于nbDims
-   foldA[]：输入，fold数组，长度等于nbDims=2
-   direction：输入，fold的模式

**返回值：**

 1. SWDNN_STATUS_SUCCESS成功

## swdnnSoftmaxBackward

```C
swdnnStatus_t swdnnSoftmaxBackward( 
	swdnnHandle_t handle, 
	swdnnSoftmaxAlgorithm_t algorithm, 
	swdnnSoftmaxMode_t mode, 
	const void *alpha, 
	const swdnnTensorDescriptor_t yDesc, 
	const void *y, 
	const swdnnTensorDescriptor_t dyDesc, 
	const void *dy, 
	const void *beta, 
	const swdnnTensorDescriptor_t dxDesc, 
	void *dx)
```

**功能描述**：

反向过程中softmax函数

**参数描述**：

-   handle：输入，控制句柄
-   algorithm：输入，softmax算法
-   mode：输入，softmax模式
-   alpha, beta：输入，扩展因子
-   yDesc：输入，y的张量描述符
-   y：输入，y的首地址指针
-   dyDesc：输入，dy的张量描述符
-   dy：输入，y的首地址指针
-   dxDesc：输入，dx的张量描述符
-   dx：输出，dx的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1. SWDNN_TENSOR_NHWC

**目前已支持的softmax算法swdnnSoftmaxAlgorithm_t**：

1.  SWDNN_SOFTMAX_ACCURATE

**目前已支持的softmax运算模式swdnnSoftmaxMode_t：**

1. SWDNN_SOFTMAX_MODE_CHANNEL

2. SWDNN_SOFTMAX_MODE_INSTANCE

## swdnnSoftmaxForward

```C
swdnnStatus_t swdnnSoftmaxForward( 
	swdnnHandle_t handle, 
	swdnnSoftmaxAlgorithm_t algorithm, 
	swdnnSoftmaxMode_t mode, 
	const void *alpha, 
	const swdnnTensorDescriptor_t xDesc, 
	const void *x, 
	const void *beta, 
	const swdnnTensorDescriptor_t yDesc, 
	void *y)
```

**功能描述**：

正向过程中softmax函数

**参数描述**：

-   handle：输入，控制句柄
-   algorithm：输入，softmax算法
-   mode：输入，softmax模式
-   alpha, beta：输入，扩展因子
-   xDesc：输入，x的张量描述符
-   x：输入，x的首地址指针
-   yDesc：输入，y的张量描述符
-   y：输出，y的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1. SWDNN_TENSOR_NHWC

**目前已支持的softmax算法swdnnSoftmaxAlgorithm_t：**

1.  SWDNN_SOFTMAX_ACCURATE

2.  SWDNN_SOFTMAX_ACCURATE_TAB

**目前已支持的softmax运算模式swdnnSoftmaxMode_t：**

1. SWDNN_SOFTMAX_MODE_CHANNEL

2. SWDNN_SOFTMAX_MODE_INSTANCE

## swdnnSquaredDifference

```C
swdnnSquaredDifference(
    swdnnHandle_t handle, 
    const swdnnTensorDescriptor_t aDesc,
    const void *A, 
    const swdnnTensorDescriptor_t bDesc,
    const void *B, 
    const swdnnTensorDescriptor_t cDesc, 
    void *C)
```

**功能描述**：

张量差平方计算 C\[i\]=(A\[i\]-B\[i\])^2，支持双向广播，即允许张量A和B的某维度为1，张量C的维度取A和B的最大维度。
比如：A(1, 16, 1, 8)，B(2, 16, 8, 8)，则C的维度为(2, 16, 8, 8)

**参数描述**：

-	handle：输入，控制句柄 
-	aDesc：输入，A的张量描述符 
-	A：输入，A的首地址指针 
-	bDesc：输入，A的张量描述符 
-	B：输入，A的首地址指针 
-	cDesc：输入，C的张量描述符 
-	C：输出，C的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC
2.  SWDNN_TENSOR_NCHW

**备注说明：**
对于SWDNN_DATA_HALF类型，要求三个张量的H、W、C三个维度的乘积为偶数，即H\*W\*C % 2 == 0

## swdnnTensorEqual

```C
swdnnTensorEqual(
    swdnnHandle_t handle, 
    const swdnnTensorDescriptor_t aDesc,
    const void *A, 
    const swdnnTensorDescriptor_t bDesc, 
    const void *B,
    const swdnnTensorDescriptor_t cDesc, void *C)

```

**功能描述**：

张量相等比较，输入与输出数据类型相同。支持双向广播，即允许张量A和B的某维度为1，张量C的维度取A和B的最大维度。
比如：A(1, 16, 1, 8)，B(2, 16, 8, 8)，则C的维度为(2, 16, 8, 8)
-	对于定点类型，则C\[i\] = (A\[i\]==B\[i\]) ? 1 : 0;
-	对于SWDNN_DATA_FLOAT类型, C\[i\] = (fabs(A\[i\]-B\[i\]) < 1e-6) ? 1.0f : 0.0f;
-	对于SWDNN_DATA_HALF类型, C\[i\] = (fabs(A\[i\]-B\[i\]) < 1e-3) ? 1.0f : 0.0f;

**参数描述**：

-	handle：输入，控制句柄 
-	aDesc：输入，A的张量描述符 
-	A：输入，A的首地址指针 
-	bDesc：输入，A的张量描述符 
-	B：输入，A的首地址指针 
-	cDesc：输入，C的张量描述符 
-	C：输出，C的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

3.  SWDNN_DATA_INT32

4.  SWDNN_DATA_INT16

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC
2.  SWDNN_TENSOR_NCHW

**其他限制**
对于SWDNN_DATA_HALF类型，要求三个张量的H、W、C三个维度的乘积为偶数，即H\*W\*C % 2 == 0

## swdnnTensorGreater

```C
swdnnTensorGreater(
    swdnnHandle_t handle, 
    const swdnnTensorDescriptor_t aDesc,
    const void *A, 
    const swdnnTensorDescriptor_t bDesc, 
    const void *B,
    const swdnnTensorDescriptor_t cDesc, void *C)

```

**功能描述**：

张量大于比较，输入与输出数据类型相同。支持双向广播，即允许张量A和B的某维度为1，张量C的维度取A和B的最大维度。
比如：A(1, 16, 1, 8)，B(2, 16, 8, 8)，则C的维度为(2, 16, 8, 8)
-	对于定点类型，则C\[i\] = (A\[i\] > B\[i\]) ? 1 : 0;
-	对于浮点类型，则C\[i\] = (A\[i\] > B\[i\]) ? 1.0f : 0.0f;

**参数描述**：

-	handle：输入，控制句柄 
-	aDesc：输入，A的张量描述符 
-	A：输入，A的首地址指针 
-	bDesc：输入，A的张量描述符 
-	B：输入，A的首地址指针 
-	cDesc：输入，C的张量描述符 
-	C：输出，C的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

3.  SWDNN_DATA_INT32

4.  SWDNN_DATA_INT16

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC
2.  SWDNN_TENSOR_NCHW

**其他限制**
对于SWDNN_DATA_HALF类型，要求三个张量的H、W、C三个维度的乘积为偶数，即H\*W\*C % 2 == 0

## swdnnTensorLess

```C
swdnnTensorLess(
    swdnnHandle_t handle, 
    const swdnnTensorDescriptor_t aDesc,
    const void *A, 
    const swdnnTensorDescriptor_t bDesc, 
    const void *B,
    const swdnnTensorDescriptor_t cDesc, void *C)

```

**功能描述**：

张量小于比较，输入与输出数据类型相同。支持双向广播，即允许张量A和B的某维度为1，张量C的维度取A和B的最大维度。
比如：A(1, 16, 1, 8)，B(2, 16, 8, 8)，则C的维度为(2, 16, 8, 8)
-	对于定点类型，则C\[i\] = (A\[i\] < B\[i\]) ? 1 : 0;
-	对于浮点类型，则C\[i\] = (A\[i\] < B\[i\]) ? 1.0f : 0.0f;

**参数描述**：

-	handle：输入，控制句柄 
-	aDesc：输入，A的张量描述符 
-	A：输入，A的首地址指针 
-	bDesc：输入，A的张量描述符 
-	B：输入，A的首地址指针 
-	cDesc：输入，C的张量描述符 
-	C：输出，C的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

3.  SWDNN_DATA_INT32

4.  SWDNN_DATA_INT16

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC
2.  SWDNN_TENSOR_NCHW

**其他限制**
对于SWDNN_DATA_HALF类型，要求三个张量的H、W、C三个维度的乘积为偶数，即H\*W\*C % 2 == 0

## swdnnSoftplusBackward

```C
swdnnStatus_t SWDNNWINAPI swdnnSoftplusBackward(
    swdnnHandle_t handle, 
    const void *threshold, 
    const void *coef,
    const void *alpha, 
    const swdnnTensorDescriptor_t dyDesc, 
    const void *dy,
    const swdnnTensorDescriptor_t xDesc, 
    const void *x, 
    const void *beta,
    const swdnnTensorDescriptor_t dxDesc, 
    void *dx);
```

**功能描述**：

反向过程中计算softplus函数

**参数描述**：

-   handle：输入，控制句柄
-   threshold：输入，指向float的指针
-   coef: 输入， 指向float的指针
-   alpha, beta：输入，扩展因子
-   yDesc：输入，y的张量描述符
-   dy：输入，dy的首地址指针
-   xDesc：输入，dx的张量描述符
-   x：输入，x的首地址指针
-   dxDesc：输出，x的张量描述符
-   dx：输出，dx的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1. SWDNN_TENSOR_NHWC
2. SWDNN_TENSOR_NCHW
3. SWDNN_TENSOR_CHWN

## swdnnSoftplusForward

```C
swdnnStatus_t SWDNNWINAPI swdnnSoftplusForward(
    swdnnHandle_t handle, const void *threshold, const void *coef,
    const void *alpha, const swdnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const swdnnTensorDescriptor_t yDesc, void *y);
```
**功能描述**：

正向过程中softplus函数

**参数描述**：

-   handle：输入，控制句柄
-   threshold：输入, 指向float的指针
-   coef：输入, 指向float的指针
-   alpha, beta：输入，扩展因子
-   xDesc：输入，x的张量描述符
-   x：输入，x的首地址指针
-   yDesc：输入，y的张量描述符
-   y：输出，y的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1. SWDNN_TENSOR_NHWC
2. SWDNN_TENSOR_NCHW
3. SWDNN_TENSOR_CHWN

## swdnnSoftsignBackward

```C
swdnnStatus_t SWDNNWINAPI swdnnSoftsignBackward(
    swdnnHandle_t handle, const void *alpha,
    const swdnnTensorDescriptor_t dyDesc, const void *dy,
    const swdnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const swdnnTensorDescriptor_t dxDesc, void *dx);
```

**功能描述**：

反向过程中计算softsign函数

**参数描述**：

-   handle：输入，控制句柄
-   alpha, beta：输入，扩展因子
-   dyDesc：输入，dy的张量描述符
-   dy：输入，dy的首地址指针
-   dxDesc：输入，dx的张量描述符
-   dx：输出，dx的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1. SWDNN_TENSOR_NHWC
2. SWDNN_TENSOR_NCHW
3. SWDNN_TENSOR_CHWN

## swdnnSoftsignForward

```C
swdnnSoftsignForward(
    swdnnHandle_t handle,
    const void *alpha, 
    const swdnnTensorDescriptor_t xDesc, 
    const void *x,
    const void *beta, 
    const swdnnTensorDescriptor_t yDesc, 
    void *y);
```
**功能描述**：

正向过程中softsign函数

**参数描述**：

-   handle：输入，控制句柄
-   alpha, beta：输入，扩展因子
-   xDesc：输入，x的张量描述符
-   x：输入，x的首地址指针
-   yDesc：输入，y的张量描述符
-   y：输出，y的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1. SWDNN_TENSOR_NHWC
2. SWDNN_TENSOR_NCHW
3. SWDNN_TENSOR_CHWN

## swdnnTransformTensor

```C
swdnnStatus_t swdnnTransformTensor( 
	swdnnHandle_t handle, 
	const void *alpha, 
	const swdnnTensorDescriptor_t xDesc, 
	const void *x, 
	const void *beta, 
	const swdnnTensorDescriptor_t yDesc, 
	void *y)
```

**功能描述**：

进行张量变换

**参数描述**：

-   handle：输入，控制句柄
-   alpha, beta：输入，扩展因子
-   xDesc：输入，x的张量描述符
-   x：输入，x的首地址指针
-   yDesc：输入，y的张量描述符
-   y：输出，y的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1. SWDNN_DATA_FLOAT

2. SWDNN_DATA_HALF

3. SWDNN_DATA_INT8

4. SWDNN_DATA_INT16

5. SWDNN_DATA_INT32

**目前已支持的数据分布swdnnTensorFormat_t：**

1. SWDNN_TENSOR_NHWC

2. SWDNN_TENSOR_NCHW

3. SWDNN_TENSOR_CHWN

## swdnnTransformTensorEx

```C
swdnnTransformTensorEx( 
	swdnnHandle_t handle, 
	const swdnnTensorTransformDescriptor_t transDesc, 
	const void *alpha, 
	const swdnnTensorDescriptor_t srcDesc, 
	const void *srcData, 
	const void *beta, 
	const swdnnTensorDescriptor_t destDesc, 
	void *destData);
```

**功能描述**：

张量的维度变换

**参数描述**：

-   handle：输入，设备句柄
-   ransDesc：输入，TensorTransform描述符
-   alpha：输入，扩展因子
-   srcDesc：输入，源张量描述符
-   srcData：输入，源张量缓冲区
-   beta：输入，扩展因子
-   destDesc：输入，结果张量描述符
-   destData：输出，结果张量缓冲区

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC

2.  SWDNN_TENSOR_NCHW

3.  SWDNN_TENSOR_CHWN

**其他限制：**

1. alpha可随机化，beta=0.

## swdnnTruncatedNormal

```C
swdnnTruncatedNormal(
	swdnnHandle_t handle, 
	unsigned long long seed, 
	float mean,
	float stddev, 
	const swdnnTensorDescriptor_t dataDesc, 
	void *data);
```

**功能描述**：

使用截断正态分布随机初始化张量

**参数描述**：

-   handle：输入，设备句柄
-   seed: 输入，随机种子
-   mean: 输入，分布的均值
-   stddev: 输入，分布的方差
-   destDesc：输入，结果张量描述符
-   destData：输出，结果张量缓冲区

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC

2.  SWDNN_TENSOR_NCHW

3.  SWDNN_TENSOR_CHWN


## swdnnwhereTensor

```C
swdnnWhereTensor(
    swdnnHandle_t handle, 
    const swdnnTensorDescriptor_t conditionDesc,
    const void *Condition, 
    const swdnnTensorDescriptor_t aDesc,
    const void *A, 
    const swdnnTensorDescriptor_t bDesc, 
    const void *B,
    const swdnnTensorDescriptor_t cDesc, 
    void *C);
```
**功能描述**：

根据条件从 X 或 Y 返回元素

**参数描述**：

-	handle：输入，控制句柄 
-   Condition: 输入，控制选择条件
-   conditionDesc: 输入，Condition的张量描述符
-	aDesc：输入，A的张量描述符 
-	A：输入，A的首地址指针 
-	bDesc：输入，B的张量描述符 
-	B：输入，B的首地址指针 
-	cDesc：输入，C的张量描述符 
-	C：输出，C的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC

2.  SWDNN_TENSOR_NCHW

## swdnnUnaryOps

```C
swdnnUnaryOps(swdnnHandle_t handle,
              swdnnUnaryOpsMode_t mode,
              const void *alpha,
              const swdnnTensorDescriptor_t xDesc,
              const void *x, 
              const swdnnTensorDescriptor_t yDesc,
              void *y)
```

**功能描述**：

根据mode选择算子

**参数描述**：

-	handle：输入，控制句柄 
-	mode: 输入，模式
-	alpha: 输入
-	xDesc：输入，x的张量描述符 
-	x：输入，x的首地址指针 
-	yDesc：输入，y的张量描述符 
-	y：输出，y的首地址指针

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC
2.  SWDNN_TENSOR_NCHW
3.  SWDNN_TENSOR_CHWN

## swdnnClampTensor

```C
swdnnClampTensor(swdnnHandle_t handle,
        float min_value,
        float max_value,
        const swdnnTensorDescriptor_t xDesc,
        void *x)
```

**功能描述**：

在x上进行截断

**参数描述**：

-	handle：输入，控制句柄 
-	min_value: 输入，截断的最小值
-	max_value: 输入，截断的最大值
-	xDesc：输入，x的张量描述符 
-	x：输入，输出，x的首地址指针 

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC
2.  SWDNN_TENSOR_NCHW
3.  SWDNN_TENSOR_CHWN

**备注：**

1. 当min_value>max_value时，所有值附max_value

## swdnnReciprocalTensor

```C
swdnnReciprocalTensor(swdnnHandle_t handle,
        const swdnnTensorDescriptor_t xDesc,
        void *x,
		const swdnnTensorDescriptor_t resultDesc,
		void *result)
```

**功能描述**：

在x上计算倒数

**参数描述**：

-	handle：输入，控制句柄 
-	xDesc：输入，x的张量描述符 
-	x：输入，输出，x的首地址指针 

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC
2.  SWDNN_TENSOR_NCHW
3.  SWDNN_TENSOR_CHWN

**备注：**

1. 未对x的值进行检查，即未对x进行非规格化数和零的检查

## swdnnTopk

```C
swdnnTopk(swdnnHandle_t handle, int axis, int k,
    const swdnnTensorDescriptor_t dataDesc,
    void *data,
    const swdnnTensorDescriptor_t resultDesc,
    void *result);
```

**功能描述**：

在axis维度上，对data数据排序，并返回前k个的索引

**参数描述**：

-	handle：输入，控制句柄 
-	axis：输入，排序的维度 
-	k：输入
-	dataDesc：输入, 输入数据描述符
-	data：输入, 输入数据首地址
-	resultDesc：输出, 输出数据描述符
-	result：输出, 输出数据首地址

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**备注：**

1. 暂时不支持axis，仅支持连续维排序，即仅支持axis==(dims-1)
2. data：4维数据, 支持float和half，并对连续维做排序（NHWC格式，对C维做排序；NCHW格式，对W做排序）
3. result：4维数据, int32类型，最后一维应为k。(如果NHWC格式，C需为k；NCHW格式，W需为k)

## swdnnConvolutionBatchNormActivationForwardInference

```C
swdnnConvolutionBatchNormActivationForwardInference(
    swdnnHandle_t handle, 
    const swdnnConvolutionDescriptor_t convDesc, 
    swdnnConvolutionFwdAlgo_t algo,
    swdnnBatchNormMode_t bn_mode,
    double epsilon,
    const swdnnActivationDescriptor_t activationDesc,
    const swdnnTensorDescriptor_t xDesc, 
    const void *x,
    const swdnnFilterDescriptor_t wDesc, 
    const void *w,
    const swdnnTensorDescriptor_t bnScaleBiasMeanVarDesc, 
    const void *bnScale,
    const void *bnBias, 
    const void *estimatedMean,
    const void *estimatedVariance, 
    const swdnnTensorDescriptor_t yDesc, 
    void *y,
    void *workSpace, 
    size_t workSpaceSizeInBytes);
```

**功能描述**：

计算convF,bnFI,actF

**参数描述**：

-	handle：输入，控制句柄 
-	convDesc：输入，卷积描述符
-	algo：输入，卷积前向算法
-	bn_mode：输入，bn模式
-	epsilon：输入
-	activationDesc：输入，激活模式
-	xDesc：输入，x描述符
-	x：输入，x的首地址指针 
-	wDesc：输入，w描述符
-	w：输入，w的首地址指针 
-	bnScaleBiasMeanVarDesc：输入，bn数据描述符
-	bnScale：输入，bnScale的首地址指针 
-	bnBias：输入，bnBias的首地址指针 
-	estimatedMean：输入，estimatedMean的首地址指针 
-	estimatedVariance：输入，estimatedVariance的首地址指针 
-	yDesc：输入，输出，y描述符
-	y：输入，输出，y的首地址指针 
-	workSpace：输入，输出
-	workSpaceSizeInBytes：输入，workSpace大小

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC
2.  SWDNN_TENSOR_NCHW
3.  SWDNN_TENSOR_CHWN

## swdnnConcat

```C
swdnnConcat(swdnnHandle_t handle,
    int axis,
    int input_length,
    swdnnTensorDescriptor_t *xDesc,
    void **x,
    swdnnTensorDescriptor_t yDesc,
    void *y)
```

**功能描述**：

执行concat操作

**参数描述**：

-	handle：输入，控制句柄 
-	axis：输入，需要拼接的维度
-	input_length：输入，拼接的数量
-	xDesc：输入描述符数组
-	x：输入，输入数据指针数组
-	yDesc：输出，输出描述符
-	y：输出，输出数据

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT
2.  SWDNN_DATA_HALF

## swdnnSplit

```C
swdnnSplit(swdnnHandle_t handle,
    int axis,
    int output_length,
    swdnnTensorDescriptor_t xDesc,
    void *x,
    swdnnTensorDescriptor_t *yDesc,
    void **y)
```

**功能描述**：

执行concat操作

**参数描述**：

-	handle：输入，控制句柄 
-	axis：输入，需要拼接的维度
-	input_length：输，拼接的数量
-	xDesc：输入，输入描述符
-	x：输入，输入数据
-	yDesc：输出描述符数组
-	y：输出，输出数据指针数组

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT
2.  SWDNN_DATA_HALF

## swdnnAbsGrad

```C
swdnnAbsGrad(swdnnHandle_t handle,
              const void *alpha,
			  const swdnnTensorDescriptor_t xDesc, const void *x,
              const swdnnTensorDescriptor_t dyDesc, const void *dy,
              const void *beta,
			  const swdnnTensorDescriptor_t dxDesc, void *dx)
```

**功能描述**：

计算abs梯度

**参数描述**：

-	handle：输入，控制句柄 
-	alpha：输入
-	xDesc：输入，输出，x的张量描述符 
-	x：输入，x的首地址指针 
-	dyDesc：输入，dy的张量描述符 
-	dy：输入，dy的首地址指针 
-	beta：输入
-	dxDesc：输入，dx的张量描述符 
-	dx：输入，dx的首地址指针 

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC
2.  SWDNN_TENSOR_NCHW
3.  SWDNN_TENSOR_CHWN

## swdnnInplaceOps

```C
swdnnInplaceOps(swdnnHandle_t handle, 
                swdnnInplaceOpsMode_t mode,
                const void *alpha, 
                const swdnnTensorDescriptor_t xDesc, 
                void *x,
                const swdnnTensorDescriptor_t yDesc, 
                void *y);
```

**功能描述**：

计算inplace算子。inplace是在自身上做计算，具体计算可以参考swdnnInplaceOpsMode_t。

**参数描述**：

-	handle：输入，控制句柄 
-	mode：输入，inplace模式
-	alpha：输入
-	xDesc：输入，输出，x的张量描述符 
-	x：输入，x的首地址指针 
-	yDesc：输入，y的张量描述符 
-	y：输入，y的首地址指针 
-	beta：输入
-	dxDesc：输入，dx的张量描述符 
-	dx：输入，dx的首地址指针 

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

2.  SWDNN_DATA_HALF

**目前已支持的数据分布swdnnTensorFormat_t：**

1.  SWDNN_TENSOR_NHWC
2.  SWDNN_TENSOR_NCHW
3.  SWDNN_TENSOR_CHWN

## swdnnNms

```C
swdnnNms(swdnnHandle_t handle,
         const swdnnTensorDescriptor_t boxesDesc,
         const float *boxes, 
         const float *score,
         const float iou_threshold, 
         int *length, 
         int *index);
```

**功能描述**：

执行非极大值抑制计算

**参数描述**：

-	handle：输入，控制句柄 
-	boxesDesc：输入，boxes描述符
-	boxes：输入，boxes的首地址
-	score：输入，score的首地址
-	iou_threshold：输入，阈值
-	length：输出，输出的长度
-	index：输出，index的索引

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT

**备注：**

1. boxes为Tensor2d，维度为（N，4）,低位存储信息顺序为：
2. score是长度为N的向量，和boxes对应，且数值应在(0,1)之间
3. iou_threshold应该在（0，1）之间
4. lengths计算输出的长度（<N）
5. index是输出长度为lengths的索引

## swdnnCumSum

```C
swdnnCumSum(const swdnnHandle_t handle,
            const int axis,
            const swdnnTensorDescriptor_t aDesc,
            const void *A,
            const swdnnTensorDescriptor_t cDesc,
            void *C);
```

**功能描述**：

执行cumSum计算

**参数描述**：

-	handle：输入，控制句柄 
-	axis：输入，需要累加的维度
-	aDesc：输入，a的描述符
-	A：输入，A的首地址
-	cDesc：输入，c的描述符
-	C：输入，C的首地址

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1.  SWDNN_DATA_FLOAT
2.  SWDNN_DATA_HALF

**备注：**

1. axis应在（0~dim-1）之间

## swdnnAdaptivePoolingForward

```C
swdnnAdaptivePoolingForward(swdnnHandle_t handle,
        swdnnAdaptivePoolingMode_t mode,
        const void *alpha,
        const swdnnTensorDescriptor_t xDesc,
        const void *x, 
        const void *beta,
        const swdnnTensorDescriptor_t yDesc,
        void *y);
```

**功能描述**：

执行自适应池化前向计算

**参数描述**：

-	handle：输入，控制句柄 
-	mode: 输入，自适应池化模式
-   alpha：输入，扩展因子
-	xDesc：输入，x的描述符
-	x：输入，X的首地址
-   beta：输入，扩展因子
-	yDesc：输入，y的描述符
-	y：输出，Y的首地址

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1. SWDNN_DATA_FLOAT
2. SWDNN_DATA_HALF

**备注：**
1. half类型存在限制（C%2==0）

## swdnnAdaptivePoolingBackward

```C
swdnnAdaptivePoolingBackward(swdnnHandle_t handle,
        swdnnAdaptivePoolingMode_t mode,
        const void *alpha,
        const swdnnTensorDescriptor_t dyDesc,
        const void *dy,
        const swdnnTensorDescriptor_t xDesc,
        const void *x, 
        const void *beta,
        const swdnnTensorDescriptor_t dxDesc,
        void *dx);
```

**功能描述**：

执行自适应池化反向计算

**参数描述**：

-	handle：输入，控制句柄 
-	mode: 输入，
-   alpha：输入，扩展因子
-	dyDesc：输入，dy的描述符
-	dy：输入，dy的首地址
-   xDesc：输入，x的描述符
-   x：输入，X的首地址
-   beta：输入，扩展因子
-	dxDesc：输入，dx的描述符
-	dx：输出，dx的首地址

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1. SWDNN_DATA_FLOAT
2. SWDNN_DATA_HALF

**备注：**
1. half类型存在限制 (C%2==0)

## swdnnGroupNormFoeward

```C
swdnnGroupNormForward(
    swdnnHandle_t handle, const int groupCount, swdnnTensorDescriptor_t xDesc,
    const void *x, swdnnTensorDescriptor_t gammaDesc, const void *gamma,
    swdnnTensorDescriptor_t betaDesc, const void *beta, double eps,
    swdnnTensorDescriptor_t yDesc, void *y, swdnnTensorDescriptor_t meanDesc,
    void *mean, swdnnTensorDescriptor_t rstdDesc, void *rstd);
```

**功能描述**：

执行

**参数描述**：

-	handle：输入，控制句柄 
-	groupCount: 输入，
-	xDesc：输入，x的描述符
-	x：输入，x的首地址
-   gammaDesc：输入，gamma的描述符
-   gamma：输入，gamma的首地址
-   betaDesc：输入，beta的描述符
-   beta：输入，beta的描述符
-   eps：输入，
-	yDesc：输入，y的描述符
-	y：输入，y的首地址
-	meanDesc：输入，mean的描述符
-	mean：输出，mean的首地址
-	rstdDesc：输入，rstd的描述符
-	rst：输出，rstd的首地址

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1. SWDNN_DATA_FLOAT
2. SWDNN_DATA_HALF

**备注：**
1. half类型存在限制((C/groupCount)%2==0)

## swdnnGroupNormBackward

```C
swdnnGroupNormBackward(
    swdnnHandle_t handle, const int groupCount, swdnnTensorDescriptor_t dyDesc, const void *dy,
    swdnnTensorDescriptor_t xDesc, const void *x, swdnnTensorDescriptor_t meanDesc,
    const void *mean, swdnnTensorDescriptor_t rstdDesc, const void *rstd,
    swdnnTensorDescriptor_t gammaDesc, const void *gamma, swdnnTensorDescriptor_t dxDesc, void *dx,
    swdnnTensorDescriptor_t dgammaDesc, void *dgamma, swdnnTensorDescriptor_t dbetaDesc,
    void *dbeta) ;
```

**功能描述**：

执行groupNorm反向计算

**参数描述**：

-	handle：输入，控制句柄 
-	groupCount: 输入，
-	dyDesc：输入，dy的描述符
-	dy：输入，dy的首地址
-	xDesc：输入，x的描述符
-	x：输入，x的首地址
-	meanDesc：输入，mean的描述符
-	mean：输入，mean的首地址
-	rstdDesc：输入，rstd的描述符
-	rst：输入，rstd的首地址
-   gammaDesc：输入，gamma的描述符
-   gamma：输入，gamma的首地址
-   dxDesc：输入，dx的描述符
-	dx：输出,dx的首地址
-   dgammaDesc：输入，dgamma的描述符
-   dgamma：输出，dgamma的首地址
-   dbetaDesc：输入，dbeta的描述符
-   dbeta：输出，dbeta的描述符

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1. SWDNN_DATA_FLOAT
2. SWDNN_DATA_HALF

**备注：**
1. half类型存在限制((C/groupCount)%2==0)

## swdnnCELossForward

```c
swdnnStatus_t SWDNNWINAPI CELossForward(
    swdnnHandle_t                           handle, 
    int                                     ignore_index, 
    swdnnLossReductionMode_t                reduction, 
    swdnnTensorDescriptor_t                 xDesc, 
    const void                              *x,
    swdnnTensorDescriptor_t                 targetDesc, 
    const void                              *target,
    swdnnTensorDescriptor_t                 wDesc, 
    const void                              *w,
    swdnnTensorDescriptor_t                 outDesc, 
    void                                    *out);
```

**功能描述**：

执行crossEntropyLoss计算，CELoss算法的具体公式如下：

a) 硬标签
$$
out  = -w_{target_n}logx_{n, y_n} (target_n !=ignore_index)\\
out=
\begin{cases}
mean(out)& \text{reduction = mean}\\
sum(out)& \text{reduction = sum}
\end{cases}
$$


b) 软标签
$$
out = -\sum_{c=0}^Cw_clogx_{n, target_n}\\
out=
\begin{cases}
mean(out)& \text{reduction = mean}\\
sum(out)& \text{reduction = sum}
\end{cases}
$$


**参数描述**：

**handle**（input）

swdnnContext的创建的句柄。查阅[swdnnHandle_t](#swdnnhandle_t)了解详情。

**ignore_index**（input）

忽略的目标值，忽略后不会对输入的梯度产生影响

**reduction**（input）

根据每个mini-batch的平均尺寸对损失进行均分或汇总。查询[swdnnLossReductionMode_t](#swdnnlossreductionmode_t)了解详情。

**xDesc**（input）

数据x的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**x**（input）：device

指向xDesc描述的数据指针。

**targetDesc**（input）

数据targetDesc的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**target**（input）：device

指向targetDesc描述的数据指针。

**wDesc**（input）

数据weights的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**w**（input）：device

指向weightsDesc描述的数据指针。

**outDesc**（input）

数据out的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**out**（output）：device

指向outDesc描述的数据指针。


**类型限制**：

<table>
	<tr>
        <th></th>
	    <th></th>
        <th>reduction</th>
	    <th>xDesc</th>
        <th>wDesc</th> 
        <th>targetDesc</th>
        <th>outDesc</th> 
	</tr >
	<tr >
        <td >1</td>
	    <td rowspan="3">硬标签</td>
        <td>NONE</td>
	    <td rowspan="6">tensor2D(float32,NC)
            tensor2D(float16,NC)</td>
        <td rowspan="6">tensor2D(float32,1C)
            tensor2D(float16,1C)</td>
        <td rowspan="3">target: tensor1D(int，N)</td>
        <td>tensor1D(float32,N)
        tensor1D(float16,N)</td>
	</tr>
    <tr >
        <td >2</td>
        <td>MEAN</td>
        <td>tensor1D(float32,1)
        tensor1D(float16,1)</td>
	</tr>
    <tr >
        <td >3</td>
        <td>SUM</td>
        <td>tensor1D(float32,1)
        tensor1D(float16,1)</td>
	</tr>
    <tr >
        <td >4</td>
        <td rowspan="3">软标签</td>
        <td>NONE</td>
        <td rowspan="3">tensor2D(float32，1C)
            tensor2D(float16,1C) </td>
        <td>tensor1D(float32,N)
        tensor1D(float16,N)</td>
	</tr>
    <tr >
        <td >5</td>
        <td>MEAN</td>
        <td>tensor1D(float32,1)
        tensor1D(float16,1)</td>
	</tr>
    <tr >
        <td >6</td>
        <td>SUM</td>
        <td>tensor1D(float32,1)
        tensor1D(float16,1)</td>
 </tr>
    </tr>
</table>

**返回值**：

* SWDNN_STATUS_SUCCESS
* SWDNN_STATUS_NOT_INITIALIZED
  * handle为空
* SWDNN_STATUS_BAD_PARAM
  * reduction不是NONE,MEAN,SUM
  * xDesc, targetDesc, wDesc,outDesc为空
  * x,target,w,out为空，或未4B对齐
  * x,target,w,out不满足Type Constraints
  * x,target,w,out维度不匹配

**示例**：

**硬标签**

```c
	const int nbDims = 2;
	int dimA[nbDims] = {N,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(xDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {1,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(weightsDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 1;
	int dimA[nbDims] = {N};
	int strideA[nbDims];
	strideA[0] = 1;
	checkSWDNN( swdnnSetTensorNdDescriptor(targetDesc, SWDNN_DATA_INT32, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {out_num,1};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(outDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );
                               
	swdnnLossReductionMode_t  reduction = 0;
	float ignore_index = 0.5;
                               
	swdnnCELossForward(handle, ignore_index, reduction, xDesc, x, targetDesc, target, wDesc, w, outDesc, out);
```



**软标签**

```c
	const int nbDims = 2;
	int dimA[nbDims] = {N,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(xDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {1,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(weightsDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {N,C};
	int strideA[nbDims];
	strideA[0] = 1;
	checkSWDNN( swdnnSetTensorNdDescriptor(targetDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {out_num,1};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(outDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );
                               
	swdnnLossReductionMode_t  reduction = 0;
	float ignore_index = 0.5;
                               
	swdnnCELossForward(handle, ignore_index, reduction, xDesc, x, targetDesc, target, wDesc, w, outDesc, out);
```


## swdnnCELossBackward

```C
swdnnStatus_t SWDNNWINAPI CELossBackward(
    swdnnHandle_t                           handle, 
    int                                     ignore_index, 
    swdnnLossReductionMode_t                reduction, 
    swdnnTensorDescriptor_t                 xDesc, 
    const void                              *x,
    swdnnTensorDescriptor_t                 targetDesc, 
    const void                              *target,
    swdnnTensorDescriptor_t                 wDesc, 
    const void                              *w,
    swdnnTensorDescriptor_t                 dyDesc, 
    void                                    *dy);
	swdnnTensorDescriptor_t                 dxDesc, 
    void                                    *dx);
```

**功能描述**：

CELossBackward是对CELossForward的反向实现。


**参数描述**：

**handle**（input）

swdnnContext的创建的句柄。查阅[swdnnHandle_t](#swdnnhandle_t)了解详情。

**ignore_index**（input）

忽略的目标值，忽略后不会对输入的梯度产生影响

**reduction**（input）

根据每个mini-batch的平均尺寸对损失进行均分或汇总。查询[swdnnLossReductionMode_t](#swdnnlossreductionmode_t)了解详情。

**xDesc**（input）

数据x的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**x**（input）：device

指向xDesc描述的数据指针。

**targetDesc**（input）

数据targetDesc的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**target**（input）：device

指向targetDesc描述的数据指针。

**wDesc**（input）

数据weights的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**w**（input）：device

指向weightsDesc描述的数据指针。

**dyDesc**（input）

数据dy的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**dy**（input）：device

指向dyDesc描述的数据指针。

**dxDesc**（input）

数据dx的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**dx**（output）：device

指向dxDesc描述的数据指针。


**类型限制**：
<table>
	<tr>
        <th></th>
	    <th></th>
        <th>reduction</th>
	    <th>x,dxDesc</th>
        <th>weightsDesc</th> 
        <th>targetDesc</th>
        <th>dyDesc</th> 
	</tr >
	<tr >
        <td >1</td>
	    <td rowspan="3">硬标签</td>
        <td>NONE</td>
	    <td rowspan="6">tensor2D(float32,NC)
            tensor2D(float16,NC)</td>
        <td rowspan="6">tensor2D(float32,1C)
            tensor2D(float16,1C)</td>
        <td rowspan="3">target: tensor1D(int，N)</td>
        <td>tensor1D(float32,N)
        tensor1D(float16,N)</td>
	</tr>
    <tr >
        <td >2</td>
        <td>MEAN</td>
        <td>tensor1D(float32,1)
        tensor1D(float16,1)</td>
	</tr>
    <tr >
        <td >3</td>
        <td>SUM</td>
        <td>tensor1D(float32,1)
        tensor1D(float16,1)</td>
	</tr>
    <tr >
        <td >4</td>
        <td rowspan="3">软标签</td>
        <td>NONE</td>
        <td rowspan="3">tensor2D(float32,1C)
            tensor2D(float16,1C) </td>
        <td>tensor1D(float32,N)
        tensor1D(float16,N)</td>
	</tr>
    <tr >
        <td >5</td>
        <td>MEAN</td>
        <td>tensor1D(float32,1)
        tensor1D(float16,1)</td>
	</tr>
    <tr >
        <td >6</td>
        <td>SUM</td>
        <td>tensor1D(float32,1)
        tensor1D(float16,1)</td>
 </tr>
    </tr>
</table>


**返回值**：
* SWDNN_STATUS_SUCCESS
* SWDNN_STATUS_NOT_INITIALIZED
  * handle为空
* SWDNN_STATUS_BAD_PARAM
  * reduction不是NONE,MEAN,SUM
  * xDesc, targetDesc, wDesc,dyDesc,dxDesc为空
  * x,target,w,dy,dx为空，或未4B对齐
  * x,target,w,dy,dx不满足Type Constraints
  * x,target,w,dy,dx维度不匹配

**示例**：
**硬标签**

```c
	const int nbDims = 2;
	int dimA[nbDims] = {N,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(xDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {1,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(weightsDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 1;
	int dimA[nbDims] = {N};
	int strideA[nbDims];
	strideA[0] = 1;
	checkSWDNN( swdnnSetTensorNdDescriptor(targetDesc, SWDNN_DATA_INT32, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {out_num,1};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(dyDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {N,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(dxDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );
                               
	swdnnLossReductionMode_t  reduction = 0;
	float ignore_index = 0.5;
                               
	swdnnCELossBackward(handle, ignore_index, reduction, xDesc, x, targetDesc, target, wDesc, w, dyDesc, dy, dxDesc, dx);
```



**软标签**

```c
	const int nbDims = 2;
	int dimA[nbDims] = {N,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(xDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {1,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(weightsDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {N,C};
	int strideA[nbDims];
	strideA[0] = 1;
	checkSWDNN( swdnnSetTensorNdDescriptor(targetDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {out_num,1};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(dyDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {N,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(dxDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );
                               
	swdnnLossReductionMode_t  reduction = 0;
	float ignore_index = 0.5;
                               
	swdnnCELossBackward(handle, ignore_index, reduction, xDesc, x, targetDesc, target, wDesc, w, dyDesc, dy, dxDesc, dx);
```


## swdnnBCELossForward

```C
swdnnStatus_t SWDNNWINAPI swdnnBCELossForward(
    swdnnHandle_t                            handle, 
    swdnnLossReductionMode_t                 mode, 
    const swdnnTensorDescriptor_t            xDesc,
    const void                               *x, 
    const swdnnTensorDescriptor_t            yDesc, 
    const void                               *y,
    const swdnnTensorDescriptor_t            wDesc, 
    const void                               *w, 
    const swdnnTensorDescriptor_t            rDesc,
    void 
```

**功能描述**：

执行BinaryCrossEnropy计算，BCELoss算法的具体公式如下：
$$
r = -w_n·[y_n ·logx_n + (1-y_n)·log(1-x_n)] \\\\
r=
\begin{cases}
mean(r)& \text{reduction = mean}\\
sum(r)& \text{reduction = sum}
\end{cases}
$$

**参数描述**：

**handle**（input）

swdnnContext的创建的句柄。查阅[swdnnHandle_t](#swdnnhandle_t)了解详情。

**mode**（input）

根据每个mini-batch的平均尺寸对损失进行均分或汇总。查询[swdnnLossReductionMode_t](#swdnnlossreductionmode_t)了解详情。

**xDesc**（input）

数据x的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**x**（input）：device

指向xDesc描述的数据指针。

**yDesc**（input）

数据yDesc的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**y**（input）：device

指向yDesc描述的数据指针。

**wDesc**（input）

数据w的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**w**（input）：device

指向wDesc描述的数据指针。

**rDesc**（input）

数据r的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**r**（output）：deviece

指向rDesc描述的数据指针。

**类型限制**：
x,y,w,r: tensor4D(float，NHWC), tensor4D(float16,NHWC) 

**返回值**：

* SWDNN_STATUS_SUCCESS
* SWDNN_STATUS_NOT_INITIALIZED
  * handle为空
  * mode不为NONE,MEAN,SUM
* SWDNN_STATUS_BAD_PARAM
  * x,y,w的format不相同或format不为NHWC
  * x,y,w没有4B对齐
* SWDNN_STATUS_INVALID_VALUE
  * x,y,w,r为空
  * x,y,w,r的维度不合法
  * x,y,w,r的维度不对应

**示例**：
```c
    checkSWDNN( swdnnSetTensor4dDescriptor(xDesc, SWDNN_TENSOR_NHWC, SWDNN_DATA_FLOAT, N, C, H,W));
	checkSWDNN( swdnnSetTensor4dDescriptor(yDesc, SWDNN_TENSOR_NHWC, SWDNN_DATA_FLOAT,  N, C, H,W));
	checkSWDNN( swdnnSetTensor4dDescriptor(wDesc, SWDNN_TENSOR_NHWC, SWDNN_DATA_FLOAT,  N, C, H,W));
	checkSWDNN( swdnnSetTensor4dDescriptor(rDesc, SWDNN_TENSOR_NHWC, SWDNN_DATA_FLOAT,  N, C, H,W));
	swdnnLossReductionMode_t  mode = 0;
	swdnnBCELossForward(handle, mode, xDesc, x, yDesc, y,wDesc, w, rDesc, r);
```
## swdnnBCELossBackward

```C
swdnnStatus_t SWDNNWINAPI swdnnBCELossBackward(
    swdnnHandle_t                            handle, 
    swdnnLossReductionMode_t                 mode, 
    const swdnnTensorDescriptor_t            xDesc,
    const void                               *x, 
    const swdnnTensorDescriptor_t            yDesc, 
    const void                               *y,
    const swdnnTensorDescriptor_t            wDesc, 
    const void                               *w, 
    const swdnnTensorDescriptor_t            dyDesc,
    void                                     *dy
    const swdnnTensorDescriptor_t            dxDesc,
    void                                     *dx);
```

**功能描述**：

BCELossBackward是对BCELossForward的反向实现。

**参数描述**：

**handle**（input）

swdnnContext的创建的句柄。查阅[swdnnHandle_t](#swdnnhandle_t)了解详情。

**mode**（input）

根据每个mini-batch的平均尺寸对损失进行均分或汇总。查询[swdnnLossReductionMode_t](#swdnnlossreductionmode_t)了解详情。

**xDesc**（input）

数据x的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**x**（input）：device

指向xDesc描述的数据指针。

**yDesc**（input）

数据yDesc的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**y**（input）：device

指向yDesc描述的数据指针。

**wDesc**（input）

数据w的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**w**（input）：device

指向wDesc描述的数据指针。

**rDesc**（input）

数据r的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**r**（output）：device

指向rDesc描述的数据指针。

**dyDesc**（input）

数据dy的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**dy**（input）：device

指向dyDesc描述的数据（device memory）指针。

**dxDesc**（input）

数据dx的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**dx**（output）：device

指向dxDesc描述的数据（device memory）指针。

**类型限制**：
x,y,w,r,dy,dx: tensor4D(float，NHWC), tensor4D(float16,NHWC) 

**返回值**：
* SWDNN_STATUS_SUCCESS
* SWDNN_STATUS_NOT_INITIALIZED
  * handle为空
* SWDNN_STATUS_BAD_PARAM
  * mode不为NONE,MEAN,SUM
  * x,y,dy,dx为空
  * x,y,dy,dx不符合Type Constraints
  * x,y,dy,dx没有4B对齐

**示例**：
```c
    checkSWDNN( swdnnSetTensor4dDescriptor(xDesc, SWDNN_TENSOR_NHWC, SWDNN_DATA_FLOAT, N, C, H,W));
	checkSWDNN( swdnnSetTensor4dDescriptor(yDesc, SWDNN_TENSOR_NHWC, SWDNN_DATA_FLOAT,  N, C, H,W));
	checkSWDNN( swdnnSetTensor4dDescriptor(wDesc, SWDNN_TENSOR_NHWC, SWDNN_DATA_FLOAT,  N, C, H,W));
	checkSWDNN( swdnnSetTensor4dDescriptor(dyDesc, SWDNN_TENSOR_NHWC, SWDNN_DATA_FLOAT,  N, C, H,W));
	checkSWDNN( swdnnSetTensor4dDescriptor(dxDesc, SWDNN_TENSOR_NHWC, SWDNN_DATA_FLOAT, N, C, H,W));
	swdnnLossReductionMode_t  mode = 0;
	swdnnBCELossBackward(handle, mode, xDesc, x, yDesc, y,wDesc, w, dyDesc, dy, dxDesc, dx);
```

## swdnnCornerPoolForward

```C
swdnnCornerPoolForward(swdnnHandle_t handle,
    swdnnCornerPoolMode_t mode,
    const swdnnTensorDescriptor_t inDesc,
    const void *input,
    const swdnnTensorDescriptor_t outDesc,
    void *output);
```

**功能描述**：

执行corner池化操作

**参数描述**：

-	handle：输入，控制句柄 
-	mode: 输入，corner池化的模式
-	inDesc：输入，input的描述符
-	in：输入，input的首地址
-	outDesc：输入，out的描述符
-	out：输出，out的首地址

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1. SWDNN_DATA_FLOAT
2. SWDNN_DATA_HALF

**备注：**
1. half类型存在限制(C需要是2的整数倍)

## swdnnCornerPoolBackward

```C
swdnnCornerPoolBackward(
    swdnnHandle_t handle,
    swdnnCornerPoolMode_t mode,
    const swdnnTensorDescriptor_t inDesc,
    const void *input,
    const swdnnTensorDescriptor_t outDesc,
    const void *output,
    const swdnnTensorDescriptor_t gradOutDesc,
    const void *grad_out,
    const swdnnTensorDescriptor_t gradInDesc,
    void *grad_in);
```

**功能描述**：

执行corner池化反向计算

**参数描述**：

-	handle：输入，控制句柄 
-	mode: 输入，corner池化的模式
-	inDesc：输入，input的描述符
-	in：输入，input的首地址
-	outDesc：输入，out的描述符
-	out：输入，out的首地址
-	gradOutDesc：输入，grad_out的描述符
-	grad_out：输入，grad_out的首地址
-	gradInDesc：输入，grad_In的描述符
-	grad_In：输出，grad_In的首地址


**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1. SWDNN_DATA_FLOAT
2. SWDNN_DATA_HALF

**备注：**
1. half类型存在限制(C需要是2的整数倍)

## swdnnTensorAny

```C
swdnnTensorAny(swdnnHandle_t handle, 
    const void *axis,
    const swdnnTensorDescriptor_t aDesc,
    const void *A,
    const swdnnTensorDescriptor_t cDesc,
    void *C);
```

**功能描述**：

沿axis维度做Any计算（所有的值做逻辑或判断）

**参数描述**：

-	handle：输入，控制句柄 
-	axis: 输入，选择的维度
-	aDesc：输入，a的描述符
-	A：输入，a的首地址
-	cDesc：输出，c的描述符
-	C：输出，c的首地址


**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1. SWDNN_DATA_FLOAT
2. SWDNN_DATA_HALF

**备注：**
1. half类型存在限制(存在部分维度需要2的整数倍)

## swdnnTensorAll

```C
swdnnTensorAll(swdnnHandle_t handle, 
    const void *axis,
    const swdnnTensorDescriptor_t aDesc,
    const void *A,
    const swdnnTensorDescriptor_t cDesc,
    void *C);
```

**功能描述**：

沿axis维度做All计算（所有的值做逻辑与判断）

**参数描述**：

-	handle：输入，控制句柄 
-	axis: 输入，选择的计算维度
-	aDesc：输入，a的描述符
-	A：输入，a的首地址
-	cDesc：输出，c的描述符
-	C：输出，c的首地址


**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1. SWDNN_DATA_FLOAT
2. SWDNN_DATA_HALF

**备注：**
1. half类型存在限制(存在部分维度需要2的整数倍)

## swdnnBernoulli

```C
swdnnBernoulli(swdnnHandle_t handle,
    unsigned long long seed,
    const swdnnTensorDescriptor_t xDesc,
    const void *x,
    const swdnnTensorDescriptor_t yDesc,
    void *y);
```

**功能描述**：

根据x产生伯努利分布


**参数描述**：

-	handle：输入，控制句柄 
-	seed: 输入，随机种子
-	xDesc：输入，x的描述符
-	x：输入，x的首地址(x需要是0~1之间)
-	yDesc：输入，y的描述符
-	y：输出，y的首地址


**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1. SWDNN_DATA_FLOAT
2. SWDNN_DATA_HALF

**备注：**
1. half类型存在限制(数据长度是2的整数倍)

## swdnnConstant

```C
swdnnConstant(const swdnnHandle_t handle,
    const void *values, const int value_len,
    const swdnnTensorDescriptor_t yDesc,
    void *y);
```

**功能描述**：

执行constant计算
value_len<=张量的总长度

**参数描述**：

-	handle：输入，控制句柄 
-	values: 输入,（数据类型和y一致）
-	value_len：输入，values的长度
-	yDesc：输出，y的描述符
-	y：输出，y的首地址

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1. SWDNN_DATA_FLOAT
2. SWDNN_DATA_HALF

**备注：**
1. half类型存在限制(value_len是2的整数倍，y是2的整数倍)

## swdnnEyeLike

```C
swdnnEyeLike(const swdnnHandle_t handle,
    const int rows, const int columns,
    const int k,
    const swdnnTensorDescriptor_t yDesc,
    void *y);
```

**功能描述**：

生成2D对角矩阵

**参数描述**：

-	handle：输入，控制句柄 
-	rows：输入
-	columns：输入
-	k: 输入, 偏移值
-	yDesc：输出，y的描述符
-	y：输出，y的首地址


**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1. SWDNN_DATA_FLOAT
2. SWDNN_DATA_HALF

**备注：**
1. half类型存在限制(列需要时2的整数倍)

## swdnnBitShift

```C
swdnnBitShift(swdnnHandle_t handle, 
	const swdnnTensorDescriptor_t aDesc,
    const void *A, 
	const swdnnTensorDescriptor_t bDesc, 
	const void *B,
    const swdnnTensorDescriptor_t cDesc,
	void *C);
```

**功能描述**：

执行移位操作
对A进行移位操作。如果B是正数，就左移；B是负数，就右移
A和B都需要是int32类型

**参数描述**：

-	handle：输入，控制句柄 
-	aDesc：输入，a的描述符
-	A：输入，A的首地址
-	bDesc：输入，b的描述符
-	B：输入，B的首地址
-	cDesc：输入，c的描述符
-	C：输出，C的首地址

**返回值**：

1. SWDNN_STATUS_SUCCESS成功

2. SWDNN_STATUS_NOT_INITIALIZED由于swdnnCreate()引发的未初始化

3. SWDNN_STATUS_ALLOC_FAILED分配内存失败

4. SWDNN_STATUS_BAD_PARAM参数错误

**目前已支持的数据类型swdnnDataType_t：**

1. SWDNN_DATA_INT32

## swdnnMSELossForward

```c
swdnnStatus_t SWDNNWINAPI swdnnMSELossForward( 
	swdnnHandle_t                              swdnnHandle, 
    swdnnLossReductionMode_t                   reduction, 
    swdnnTensorDescriptor_t                    inputDesc, 
    const void                                 *input, 
    swdnnTensorDescriptor_t                    targetDesc, 
    const void                                 *target, 
    swdnnTensorDescriptor_t                    outputDesc, 
    void                                       *output )
```

**功能描述**：

MSELoss的计算公式为：
$$
out = (input_{n,c}-target_{n,c})^2
\\out=
\begin{cases}
mean(out)& \text{reduction = mean}\\
sum(out)& \text{reduction = sum}
\end{cases}
$$	

**参数描述**：

**handle**（input）

swdnnContext的创建的句柄。查阅[swdnnHandle_t](#swdnnhandle_t)了解详情。

**reduction**（input）

根据每个mini-batch的平均尺寸对损失进行均分或汇总。查询[swdnnLossReductionMode_t](#swdnnlossreductionmode_t)了解详情。

**inputDesc**（input）

数据input的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**input**（input）：device

指向inputDesc描述的数据指针。

**targetDesc**（input）

数据targetDesc的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**target**（input）：device

指向targetDesc描述的数据指针。

**outDesc**（input）

数据out的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**out**（output）：device

指向outDesc描述的数据指针。

**类型限制**：

input,target,out: tensor4D(float32，NHWC), tensor4D(float32，NCHW),tensor4D(float16,NHWC) ,tensor4D(float16，NCHW)

**返回值**：

* SWDNN_STATUS_SUCCESS
* SWDNN_STATUS_NOT_INITIALIZED
  * handle为空
* SWDNN_STATUS_BAD_PARAM
  * mode不为NONE,MEAN,SUM
  * inputDesc,targetDesc,outDesc为空或不合法
  * inputDesc,targetDesc,outDesc不对应
  * input,target,out为空
  * input,target,out不符合Type Constraints
  * input,target,out没有4B对齐

**示例**：

```C
	checkSWDNN( swdnnSetTensor4dDescriptor(inputDesc, SWDNN_TENSOR_NHWC, SWDNN_DATA_FLOAT, N, C, H, W));
	checkSWDNN( swdnnSetTensor4dDescriptor(targetDesc, SWDNN_TENSOR_NHWC, SWDNN_DATA_FLOAT,  N, C, H, W));
	checkSWDNN( swdnnSetTensor4dDescriptor(outputDesc, SWDNN_TENSOR_NHWC, SWDNN_DATA_FLOAT,  N, C, H, W));

	swdnnLossReductionMode_t  reduction = 0;

	swdnnMSELossForward(  handle, reduction, inputDesc, input, targetDesc, target, outputDesc, output )
```

## swdnnMSELossBackward

```c
swdnnStatus_t SWDNNWINAPI swdnnMSELossBackward( 
    swdnnHandle_t                               swdnnHandle, 
    swdnnLossReductionMode_t                    reduction, 
    swdnnTensorDescriptor_t                     inputDesc, 
    const void                                  *input, 
    swdnnTensorDescriptor_t                     targetDesc, 
    const void                                  *target, 
    swdnnTensorDescriptor_t                     doutputDesc, 
    const void                                  *doutput, 
    swdnnTensorDescriptor_t                     dinputDesc, 
    void                                        *dinput, 
    swdnnTensorDescriptor_t                     dtargetDesc, 
    void                                        *dtarget)
```

**功能描述**：

MSELossBackward是对MSELossForward的反向实现。

**参数描述**：

**handle**（input）

swdnnContext的创建的句柄。查阅[swdnnHandle_t](#swdnnhandle_t)了解详情。

**reduction**（input）

根据每个mini-batch的平均尺寸对损失进行均分或汇总。查询[swdnnLossReductionMode_t](#swdnnlossreductionmode_t)了解详情。

**inputDesc**（input）

数据input的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**input**（input）：device

指向inputDesc描述的数据指针。

**targetDesc**（input）

数据targetDesc的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**target**（input）：device

指向targetDesc描述的数据指针。

**doutDesc**（input）

数据dout的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**dout**（input）：device

指向doutDesc描述的数据指针。

**dinputDesc**（input）

数据dinput的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**dinput**（output）：device

指向dinputDesc描述的数据指针。

**类型限制**：

input,target,dout,dinput: tensor4D(float32，NHWC)tensor4D(float16,NHWC) 

**返回值**：

* SWDNN_STATUS_SUCCESS
* SWDNN_STATUS_NOT_INITIALIZED
  * handle为空
* SWDNN_STATUS_BAD_PARAM
  * mode不为NONE,MEAN,SUM
  * inputDesc,targetDesc,doutDesc,dinputDesc为空或不合法
  * inputDesc,targetDesc,doutDesc,dinputDesc不对应
  * input,target,dout,dinput为空
  * input,target,dout,dinput不符合Type Constraints
  * input,target,dout,dinput没有4B对齐

**示例**：

```c
	checkSWDNN( swdnnSetTensor4dDescriptor(inputDesc, SWDNN_TENSOR_NHWC, SWDNN_DATA_FLOAT, N, C, H, W));
	checkSWDNN( swdnnSetTensor4dDescriptor(targetDesc, SWDNN_TENSOR_NHWC, SWDNN_DATA_FLOAT,  N, C, H, W));
	checkSWDNN( swdnnSetTensor4dDescriptor(doutputDesc, SWDNN_TENSOR_NHWC, SWDNN_DATA_FLOAT,  N, C, H, W));
	checkSWDNN( swdnnSetTensor4dDescriptor(dinputDesc, SWDNN_TENSOR_NHWC, SWDNN_DATA_FLOAT, N, C, H, W));
	checkSWDNN( swdnnSetTensor4dDescriptor(dtargetDesc, SWDNN_TENSOR_NHWC, SWDNN_DATA_FLOAT,  N, C, H, W));

	swdnnLossReductionMode_t  reduction = 0;

	swdnnMSELossBackward(  handle, reduction, inputDesc, input, targetDesc, target, doutputDesc, doutput, 
                        dinputDesc, dinput, dtargetDesc, dtarget);
```
## swdnnSigmoidBCELossForward

```c
swdnnStatus_t SWDNNWINAPI swdnnSigmoidBCELossForward(
	swdnnHandle_t                         handle, 
    swdnnLossReductionMode_t              reduction,
	swdnnTensorDescriptor_t               xDesc, 
    const void                            *x,
	swdnnTensorDescriptor_t               targetDesc, 
    const void                            *target,
	swdnnTensorDescriptor_t               weightsDesc, 
    const void                            *weights,
	swdnnTensorDescriptor_t               pos_weightsDesc, 
    const void                            *pos_weights,
	swdnnTensorDescriptor_t               outDesc, 
    void                                  *out);
```

**功能描述**：

SigmoidBCELoss算法的具体公式如下：
$$
out = -w_{n,c}·[pos\_weight_cy_{n,c}·logσ(x_{n,c})+(1-target_{n,c})·log(1-σ(x_{n,c}))] \\\\
out=
\begin{cases}
mean(out)& \text{reduction = mean}\\
sum(out)& \text{reduction = sum}
\end{cases}
$$

**参数描述**：

**handle**（input）

swdnnContext的创建的句柄。查阅[swdnnHandle_t](#swdnnhandle_t)了解详情。

**mode**（input）

根据每个mini-batch的平均尺寸对损失进行均分或汇总。查询[swdnnLossReductionMode_t](#swdnnlossreductionmode_t)了解详情。

**xDesc**（input）

数据x的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**x**（input）：device

指向xDesc描述的数据指针。

**targetDesc**（input）

数据targetDesc的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**target**（input）

指向targetDesc描述的数据指针。

**weightsDesc**（input）

数据weights的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**weights**（input）：device

指向weightsDesc描述的数据指针。

**pos_weightsDesc**（input）

数据pos_weights的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**pos_weights**（input）：device

指向pos_weightsDesc描述的数据指针。

**outDesc**（input）

数据out的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**out**（output）：deviece

指向outDesc描述的数据指针。

**类型限制**：

x，target, out: tensor2D(float，NC), tensor2D(float16, NC) 

weights,pos_weights: tensor1D(float，C), tensor1D(float16, C) 

**返回值**：

* SWDNN_STATUS_SUCCESS
* SWDNN_STATUS_NOT_INITIALIZED
  * handle为空
* SWDNN_STATUS_BAD_PARAM
  * xDesc,targetDesc,outDesc为空
  * x,target,out为空
  * reduction不为NONE,MEAN,SUM
  * x,target,w,pos_w不符合Type Constraints
  * x和target的维度不对应

**示例**：

```c
	const int nbDims = 2;
	int dimA[nbDims] = {N,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	swdnnSetTensorNdDescriptor(xDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA);

	const int nbDims = 2;
	int dimA[nbDims] = {N,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	swdnnSetTensorNdDescriptor(targetDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA);

	const int nbDims = 1;
	int dimA[nbDims] = {C};
	int strideA[nbDims];
	strideA[0] = 2;
	swdnnSetTensorNdDescriptor(weightsDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA);

	const int nbDims = 1;
	int dimA[nbDims] = {C};
	int strideA[nbDims];
	strideA[0] = 2;
	swdnnSetTensorNdDescriptor(pow_weightsDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA);

	const int nbDims = 2;
	int dimA[nbDims] = {N,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	swdnnSetTensorNdDescriptor(outDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA);
                               
	swdnnLossReductionMode_t  mode = 0;
                               
	swdnnSigmoidBCELossForward(swdnnHandle, mode, xDesc, x, targetDesc, target, weightsDesc, weights, pos_weightsDesc, pos_weights, outDesc, out);
```


## swdnnSigmoidBCELossBackward

```c
swdnnStatus_t SWDNNWINAPI  swdnnSigmoidBCELossBackward(
    swdnnHandle_t                         handle, 
    swdnnLossReductionMode_t              reduction, 
    swdnnTensorDescriptor_t               xDesc, 
    const void                            *x,
    swdnnTensorDescriptor_t               targetDesc, 
    const void                            *target,
    swdnnTensorDescriptor_t               wDesc, 
    const void                            *w,
    swdnnTensorDescriptor_t               pos_wDesc, 
    const void                            *pos_w,
    swdnnTensorDescriptor_t               dyDesc, 
    const void                            *dy,
    swdnnTensorDescriptor_t               dxDesc, 
    void                                  *dx);
```

**功能描述**：

SigmoidBCELossBackward是对SigmoidBCELossForward的反向实现。

**参数描述**：

**handle**（input）

swdnnContext的创建的句柄。查阅[swdnnHandle_t](#swdnnhandle_t)了解详情。

**mode**（input）

根据每个mini-batch的平均尺寸对损失进行均分或汇总。查询[swdnnLossReductionMode_t](#swdnnlossreductionmode_t)了解详情。

**xDesc**（input）

数据x的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**x**（input）：device

指向xDesc描述的数据指针。

**targetDesc**（input）

数据targetDesc的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**target**（input）

指向targetDesc描述的数据指针。

**wDesc**（input）

数据w的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**w**（input）：device

指向wDesc描述的数据指针。

**pos_wDesc**（input）

数据pos_w的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**pos_w（input）：device

指向pos_wDesc描述的数据指针。

**dyDesc**（input）

数据dy的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**dy**（input）：deviece

指向dyDesc描述的数据指针。

**dxDesc**（input）

数据dx的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**dx**（output）：deviece

指向dxDesc描述的数据指针。

**类型限制**：

x，target,  dy, dx: tensor2D(float，NC), tensor2D(float16, NC) 

weights,pos_weights: tensor1D(float，C), tensor1D(float16, C) 

**返回值**：

* SWDNN_STATUS_SUCCESS
* SWDNN_STATUS_NOT_INITIALIZED
  * handle为空
* SWDNN_STATUS_BAD_PARAM
  * xDesc,targetDesc,outDesc为空
  * x,target,out为空
  * reduction不为NONE,MEAN,SUM
  * x,target,w,pos_w不符合Type Constraints
  * x和target的维度不对应

**示例**：

```c
    const int nbDims = 2;
	int dimA[nbDims] = {N,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	swdnnSetTensorNdDescriptor(xDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA);

	const int nbDims = 2;
	int dimA[nbDims] = {N,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	swdnnSetTensorNdDescriptor(targetDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA);

	const int nbDims = 1;
	int dimA[nbDims] = {C};
	int strideA[nbDims];
	strideA[0] = 2;
	swdnnSetTensorNdDescriptor(wsDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA);

	const int nbDims = 1;
	int dimA[nbDims] = {C};
	int strideA[nbDims];
	strideA[0] = 2;
	swdnnSetTensorNdDescriptor(pow_wDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA);

	const int nbDims = 2;
	int dimA[nbDims] = {N,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	swdnnSetTensorNdDescriptor(dyDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA);

	const int nbDims = 2;
	int dimA[nbDims] = {N,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	swdnnSetTensorNdDescriptor(dxDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA);
                               
	swdnnLossReductionMode_t  mode = 0;
                               
	swdnnSigmoidBCELossBackward(swdnnHandle, mode, xDesc, x, targetDesc, target, wDesc, w, pos_wDesc, pos_w, dyDesc, dy, dxDesc, dx);
```


## swdnnSoftmaxCELossForward

```c
swdnnStatus_t SWDNNWINAPI swdnnSoftmaxCELossForward(
    swdnnHandle_t                           handle, 
    int                                     ignore_index, 
    swdnnLossReductionMode_t                reduction, 
    float                                   label_smoothing,
    swdnnTensorDescriptor_t                 xDesc, 
    const void                              *x,
    swdnnTensorDescriptor_t                 targetDesc, 
    const void                              *target,
    swdnnTensorDescriptor_t                 wDesc, 
    const void                              *w,
    swdnnTensorDescriptor_t                 outDesc, 
    void                                    *out);
```

**功能描述**：

SoftmaxCELoss算法的具体公式如下：

a) 硬标签
$$
out  = -w_{target_n}log\frac{exp(x_{n, target_n})} {\sum_{c =1}^C{exp(x_{n, c})}}⋅1 (target_n !=ignore_index)
\\out=
\begin{cases}
mean(out)& \text{reduction = mean}\\
sum(out)& \text{reduction = sum}
\end{cases}
$$


b) 软标签
$$
out = -\sum_{c=0}^Cw_clog\frac{exp(x_{n,c})} {\sum_{c =1}^C{exp(x_{n, i})}}target_{n,c}\\
out=
\begin{cases}
mean(out)& \text{reduction = mean}\\
sum(out)& \text{reduction = sum}
\end{cases}
$$

​		

**参数描述**：

**handle**（input）

swdnnContext的创建的句柄。查阅[swdnnHandle_t](#swdnnhandle_t)了解详情。

**ignore_index**（input）

忽略的目标值，忽略后不会对输入的梯度产生影响

**reduction**（input）

根据每个mini-batch的平均尺寸对损失进行均分或汇总。查询[swdnnLossReductionMode_t](#swdnnlossreductionmode_t)了解详情。

**label_smoothing**（input）

计算损失时的平滑量，范围为[0.0,1.0]。

**xDesc**（input）

数据x的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**x**（input）：device

指向xDesc描述的数据指针。

**targetDesc**（input）

数据targetDesc的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**target**（input）：device

指向targetDesc描述的数据指针。

**wDesc**（input）

数据weights的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**w**（input）：device

指向weightsDesc描述的数据指针。

**outDesc**（input）

数据out的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**out**（output）：device

指向outDesc描述的数据指针。

**类型限制**：

<table>
	<tr>
        <th></th>
	    <th></th>
        <th>reduction</th>
	    <th>xDesc</th>
        <th>wDesc</th> 
        <th>targetDesc</th>
        <th>outDesc</th> 
	</tr >
	<tr >
        <td >1</td>
	    <td rowspan="3">硬标签</td>
        <td>NONE</td>
	    <td rowspan="6">tensor2D(float32,NC)
            tensor2D(float16,NC)</td>
        <td rowspan="6">tensor2D(float32,1C)
            tensor2D(float16,1C)</td>
        <td rowspan="3">target: tensor1D(int，N)</td>
        <td>tensor1D(float32,N)
        tensor1D(float16,N)</td>
	</tr>
    <tr >
        <td >2</td>
        <td>MEAN</td>
        <td>tensor1D(float32,1)
        tensor1D(float16,1)</td>
	</tr>
    <tr >
        <td >3</td>
        <td>SUM</td>
        <td>tensor1D(float32,1)
        tensor1D(float16,1)</td>
	</tr>
    <tr >
        <td >4</td>
        <td rowspan="3">软标签</td>
        <td>NONE</td>
        <td rowspan="3">tensor2D(float32，1C)
            tensor2D(float16,1C) </td>
        <td>tensor1D(float32,N)
        tensor1D(float16,N)</td>
	</tr>
    <tr >
        <td >5</td>
        <td>MEAN</td>
        <td>tensor1D(float32,1)
        tensor1D(float16,1)</td>
	</tr>
    <tr >
        <td >6</td>
        <td>SUM</td>
        <td>tensor1D(float32,1)
        tensor1D(float16,1)</td>
 </tr>
    </tr>
</table>

**返回值**：

* SWDNN_STATUS_SUCCESS
* SWDNN_STATUS_NOT_INITIALIZED
  * handle为空
* SWDNN_STATUS_BAD_PARAM
  * reduction不是NONE,MEAN,SUM
  * xDesc, targetDesc, wDesc,outDesc为空
  * x,target,w,out为空，或未4B对齐
  * x,target,w,out不满足Type Constraints
  * x,target,w,out维度不匹配

**示例**：

**硬标签**

```c
	const int nbDims = 2;
	int dimA[nbDims] = {N,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(xDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {1,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(weightsDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 1;
	int dimA[nbDims] = {N};
	int strideA[nbDims];
	strideA[0] = 1;
	checkSWDNN( swdnnSetTensorNdDescriptor(targetDesc, SWDNN_DATA_INT32, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {outNum,1};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(outDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );
                               
	swdnnLossReductionMode_t  reduction = 0;
	float ignore_index = 0.5;
	float label_smoothing = 0.5;
                               
	swdnnSoftmaxCELossForward(handle, ignore_index, reduction, label_smoothing, xDesc, x, targetDesc, target, wDesc, w, outDesc, out);
```



**软标签**

```c
	const int nbDims = 2;
	int dimA[nbDims] = {N,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(xDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {1,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(weightsDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {N,C};
	int strideA[nbDims];
	strideA[0] = 1;
	checkSWDNN( swdnnSetTensorNdDescriptor(targetDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {outNum,1};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(outDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );
                               
	swdnnLossReductionMode_t  reduction = 0;
	float ignore_index = 0.5;
	float label_smoothing = 0.5;
                               
	swdnnSoftmaxCELossForward(handle, ignore_index, reduction, label_smoothing, xDesc, x, targetDesc, target, wDesc, w, outDesc, out);
```


## swdnnSoftmaxCELossBackward

```c
swdnnStatus_t SWDNNWINAPI swdnnSoftmaxCELossBackward(
    swdnnHandle_t                           handle, 
    int                                     ignore_index, 
    swdnnLossReductionMode_t                reduction, 
    float                                   label_smoothing,
    swdnnTensorDescriptor_t                 xDesc, 
    const void                              *x,
    swdnnTensorDescriptor_t                 targetDesc, 
    const void                              *target,
    swdnnTensorDescriptor_t                 wDesc, 
    const void                              *w,
    swdnnTensorDescriptor_t                 dyDesc, 
    void                                    *dy);
	swdnnTensorDescriptor_t                 dxDesc, 
    void                                    *dx);
```

**功能描述**：

SoftmaxCELossBackward是对SoftmaxCELossForward的反向实现。

**参数描述**：

**handle**（input）

swdnnContext的创建的句柄。查阅[swdnnHandle_t](#swdnnhandle_t)了解详情。

**ignore_index**（input）

忽略的目标值，忽略后不会对输入的梯度产生影响

**reduction**（input）

根据每个mini-batch的平均尺寸对损失进行均分或汇总。查询[swdnnLossReductionMode_t](#swdnnlossreductionmode_t)了解详情。

**label_smoothing**（input）

计算损失时的平滑量，范围为[0.0,1.0]。

**xDesc**（input）

数据x的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**x**（input）：device

指向xDesc描述的数据指针。

**targetDesc**（input）

数据targetDesc的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**target**（input）：device

指向targetDesc描述的数据指针。

**wDesc**（input）

数据weights的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**w**（input）：device

指向weightsDesc描述的数据指针。

**dyDesc**（input）

数据dy的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**dy**（input）：device

指向dyDesc描述的数据指针。

**dxDesc**（input）

数据dx的描述符。查阅[swdnnTensorDescriptor_t](#swdnntensordescriptor_t)了解详情。

**dx**（output）：device

指向dxDesc描述的数据指针。

**类型限制**：

<table>
	<tr>
        <th></th>
	    <th></th>
        <th>reduction</th>
	    <th>x,dxDesc</th>
        <th>weightsDesc</th> 
        <th>targetDesc</th>
        <th>dyDesc</th> 
	</tr >
	<tr >
        <td >1</td>
	    <td rowspan="3">硬标签</td>
        <td>NONE</td>
	    <td rowspan="6">tensor2D(float32,NC)
            tensor2D(float16,NC)</td>
        <td rowspan="6">tensor2D(float32,1C)
            tensor2D(float16,1C)</td>
        <td rowspan="3">target: tensor1D(int，N)</td>
        <td>tensor1D(float32,N)
        tensor1D(float16,N)</td>
	</tr>
    <tr >
        <td >2</td>
        <td>MEAN</td>
        <td>tensor1D(float32,1)
        tensor1D(float16,1)</td>
	</tr>
    <tr >
        <td >3</td>
        <td>SUM</td>
        <td>tensor1D(float32,1)
        tensor1D(float16,1)</td>
	</tr>
    <tr >
        <td >4</td>
        <td rowspan="3">软标签</td>
        <td>NONE</td>
        <td rowspan="3">tensor2D(float32,1C)
            tensor2D(float16,1C) </td>
        <td>tensor1D(float32,N)
        tensor1D(float16,N)</td>
	</tr>
    <tr >
        <td >5</td>
        <td>MEAN</td>
        <td>tensor1D(float32,1)
        tensor1D(float16,1)</td>
	</tr>
    <tr >
        <td >6</td>
        <td>SUM</td>
        <td>tensor1D(float32,1)
        tensor1D(float16,1)</td>
 </tr>
    </tr>
</table>

**返回值**：

* SWDNN_STATUS_SUCCESS
* SWDNN_STATUS_NOT_INITIALIZED
  * handle为空
* SWDNN_STATUS_BAD_PARAM
  * reduction不是NONE,MEAN,SUM
  * xDesc, targetDesc, wDesc,dyDesc,dxDesc为空
  * x,target,w,dy,dx为空，或未4B对齐
  * x,target,w,dy,dx不满足Type Constraints
  * x,target,w,dy,dx维度不匹配

**示例**：

**硬标签**

```c
	const int nbDims = 2;
	int dimA[nbDims] = {N,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(xDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {1,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(weightsDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 1;
	int dimA[nbDims] = {N};
	int strideA[nbDims];
	strideA[0] = 1;
	checkSWDNN( swdnnSetTensorNdDescriptor(targetDesc, SWDNN_DATA_INT32, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {out_num,1};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(dyDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {N,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(dxDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );
                               
	swdnnLossReductionMode_t  reduction = 0;
	float ignore_index = 0.5;
	float label_smoothing = 0.5;
                               
	swdnnSoftmaxCELossBackward(handle, ignore_index, reduction, xDesc, x, targetDesc, target, wDesc, w, dyDesc, dy, dxDesc, dx);
```



**软标签**

```c
	const int nbDims = 2;
	int dimA[nbDims] = {N,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(xDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {1,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(weightsDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {N,C};
	int strideA[nbDims];
	strideA[0] = 1;
	checkSWDNN( swdnnSetTensorNdDescriptor(targetDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {out_num,1};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(dyDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );

	const int nbDims = 2;
	int dimA[nbDims] = {N,C};
	int strideA[nbDims];
	strideA[1] = 1; strideA[0] = dimA[1];
	checkSWDNN( swdnnSetTensorNdDescriptor(dxDesc, SWDNN_DATA_FLOAT, nbDims, dimA, strideA) );
                               
	swdnnLossReductionMode_t  reduction = 0;
	float ignore_index = 0.5;
	float label_smoothing = 0.5;
                               
	swdnnSoftmaxCELossBackward(handle, ignore_index, reduction, xDesc, x, targetDesc, target, wDesc, w, dyDesc, dy, dxDesc, dx);
```

