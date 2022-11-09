# 概况

The Tecorigin Basic Linear Algebra Subprograms(TecoBLAS)是一个面向SW-AI芯片的基础线性代数程序集。TecoBLAS旨在提供标准BLAS例程子集的高度优化实现，目前正在迭代开发中，已经实现了部分AI场景常用的算子，集成了江南所开发优化的单精度和半精度GEMM接口。
## 版本历史

| 文档名称 | TecoBLAS 参考手册                     |
|----------|--------------------------------------|
| 版本     | V1.1.0                                |
| 作者     | High Performance Engineering （HPE） |
| 时间     | 2022.10.12                       |

<a href="../_images/tecoblas-v1.0.0.pdf" target="_blank">离线版本文档</a>

## 更新记录

### v1.1.0更新记录

- 添加Hgemm（偶数）支持 half 
- 添加Sgemm 支持 float
- 添加gemv 支持 half， float，但不支持转置参数
- 修复copy swap 中的bug 支持 half， float
- 添加tril triu 支持 float
- 添加tril triu API 说明
- 去除Hgemm在半精度情况下，M, N, K, lda, ldb, ldc是2的整数倍限制

# 数据类型参考

用于描述TecoBLAS库API中使用到的所有枚举变量和结构体类型。

## tblasHandle_t
tblasHandle_t是一个结构体指针类型，用于保存上下文信息。

## tblasPropertyType_t

tblasPropertyType_t是枚举变量，用于获取版本发布的属性。

1. TBLAS_MAJOR 

2. TBLAS_MINOR

3. TBLAS_PATCH

## tblasStatus_t

tblasStatus_t是枚举变量，用于描述库函数的**返回值**。

1. TBLAS_STATUS_SUCCESS 成功

2. TBLAS_STATUS_NOT_INITIALIZED 未初始化handle

3. TBLAS_STATUS_ALLOC_FAILED 空间分配失败

4. TBLAS_STATUS_BAD_PARAM 参数错误

5. TBLAS_STATUS_INTERNAL_ERROR 不可预知错误

6. TBLAS_STATUS_INVALID_VALUE 无效参数

7. TBLAS_STATUS_ARCH_MISMATCH 架构不匹配

8. TBLAS_STATUS_MAPPING_ERROR 语言合并错误

9. TBLAS_STATUS_EXECUTION_FAILED 执行失败

10.  TBLAS_STATUS_NOT_SUPPORTED 目前不支持

11.  TBLAS_STATUS_CONFIG_ERROR 配置错误，一般指STASK_CORE_NUM环境变量没有配置为32

12. TBLAS_STATUS_BAD_LD leading dimension类参数错误


## tblasOperation_t

tblasOperation_t是枚举类型，用于选择转置操作类型。

1. TBLAS_OP_N 不转置

2. TBLAS_OP_T 转置

3. TBLAS_OP_C 共轭转置


# 辅助 API 列表

## tblasCreate

```C
tblasStatus_t TBLASWINAPI tblasCreate(
    tblasHandle_t *handle);
```
**功能描述**：

创建句柄

**参数描述：**

-   handle，输入，句柄的指针

**返回值：**

1. TBLAS_STATUS_SUCCESS 成功
2. TBLAS_STATUS_CONFIG_ERROR 环境变量STASK_CORE_NUM设置错误，应该为32

## tblasDestroy

```C
tblasStatus_t TBLASWINAPI tblasDestroy(
    tblasHandle_t handle);
```
**功能描述**：

销毁句柄

**参数描述：**

-   handle，输入，句柄

**返回值：**

1. TBLAS_STATUS_SUCCESS 成功
2. TBLAS_STATUS_BAS_PARAM 参数错误，handle为空

## tblasGetErrorString

```C
const char *tblasGetErrorString(
    tblasStatus_t status);
```
**功能描述**：

根据错误码，返回错误字符串表示

**参数描述：**

-   status，输入，错误码

**返回值：**

1. 错误字符串表示
2. "unknown error"

## tblasGetVersion

```C
tblasStatus_t TBLASWINAPI tblasGetVersion(
    tblasHandle_t handle, 
    int *version);
```
**功能描述**：

获得版本号

**参数描述：**

-   handle，输入，句柄
-   version，输出，版本号

**返回值：**

1. TBLAS_STATUS_SUCCESS 成功

## tblasGetProperty

```C
tblasStatus_t TBLASWINAPI tblasGetProperty(
    tblasPropertyType_t propertyType,
    int *value);
```
**功能描述**：

根据属性类型，获取属性，包括Major、Minor、Patch等。

**参数描述：**

-   PropertyType，输入，属性名称
-   value，输出，属性值

**返回值：**

1. TBLAS_STATUS_SUCCESS 成功
2. TBLAS_STATUS_BAD_PARAM 属性参数错误

# Level-1 API 列表

## tblasI?amax

```C
tblasStatus_t TBLASWINAPI tblasISamax(
    tblasHandle_t handle, 
    int n, 
    const float *x,
    int incx, 
    int *result);
tblasStatus_t TBLASWINAPI tblasIHamax(
    tblasHandle_t handle, 
    int n, 
    const half *x,
    int incx, 
    int *result);
```
**功能描述**：

finds the (smallest) index of the element of the maximum magnitude.

**支持的数据类型**
- float
- half

**参数描述：**

 -  handle, <font color=#008000> input</font>, handle to the TecoBLAS library
 -  n, <font color=#008000> input</font>, the length of the vector x
 -  x, <font color=#008000> input</font>, the vector x
 -  incx, <font color=#008000> input</font>, the increment of the vector x
 -  result, <font color=#FF0000> output</font>, the (smallest) index of the element of the maximum magnitude

**返回值：**

1. TBLAS_STATUS_SUCCESS 成功
2. TBLAS_STATUS_BAD_PARAM 参数错误
3. TBLAS_STATUS_NOT_SUPPORTED 目前不支持
4. TBLAS_STATUS_NOT_INITIALIZED handle 未初始化


## tblasI?amin

```C
tblasStatus_t TBLASWINAPI tblasISamin(
    tblasHandle_t handle, 
    int n, 
    const float *x,
    int incx, 
    int *result);
tblasStatus_t TBLASWINAPI tblasIHamin(
    tblasHandle_t handle, 
    int n, 
    const half *x,
    int incx, 
    int *result);
```
**功能描述**：

finds the (smallest) index of the element of the minimum magnitude.

**支持的数据类型**
- float
- half

**参数描述：**

 -  handle, <font color=#008000> input</font>, handle to the TecoBLAS library
 -  n, <font color=#008000> input</font>, the length of the vector x
 -  x, <font color=#008000> input</font>, the vector x
 -  incx, <font color=#008000> input</font>, the increment of the vector x
 -  result, <font color=#FF0000> output</font>, the (smallest) index of the element of the minimum magnitude

**返回值：**

1. TBLAS_STATUS_SUCCESS 成功
2. TBLAS_STATUS_BAD_PARAM 参数错误
3. TBLAS_STATUS_NOT_SUPPORTED 目前不支持
4. TBLAS_STATUS_NOT_INITIALIZED handle 未初始化


## tblas?asum
```C
tblasStatus_t TBLASWINAPI tblasSasum(
    tblasHandle_t handle, 
    int n, 
    const float *x,
    int incx, 
    float *result);
tblasStatus_t TBLASWINAPI tblasHasum(
    tblasHandle_t handle,
    int n,
    const half *x,
    int incx,
    float *result);
```

**功能描述**：

computes the sum of the absolute values of the elements of vector x

**支持的数据类型**
- float
- half

**参数描述：**

- handle, <font color=#008000> input</font>, handle to the TecoBLAS library
- n, <font color=#008000> input</font>, the length of the vector x
- x, <font color=#008000> input</font>, the vector x
- incx, <font color=#008000> input</font>, the increment of the vector x
- result, <font color=#FF0000> output</font>, the absolute sum of the elements of vector x

**返回值：**

1. TBLAS_STATUS_SUCCESS 成功
2. TBLAS_STATUS_BAD_PARAM 参数错误
3. TBLAS_STATUS_NOT_SUPPORTED 目前不支持
4. TBLAS_STATUS_NOT_INITIALIZED handle 未初始化


## tblas?axpy
```C
tblasStatus_t TBLASWINAPI tblasSaxpy(
    tblasHandle_t handle, 
    int n, 
    float alpha,
    const float *x, 
    int incx, 
    float *y, 
    int incy);
tblasStatus_t TBLASWINAPI tblasHaxpy(
    tblasHandle_t handle, 
    int n, 
    half alpha,
    const half *x, 
    int incx, 
    half *y, 
    int incy);
```

**功能描述**：

multiplies the vector x by the scalar alpha and adds it to the vector y

**支持的数据类型**
- float
- half

**参数描述：**
- handle, <font color=#008000> input</font>, handle to the TecoBLAS library
- n, <font color=#008000> input</font>, the length of the vector x
- x, <font color=#008000> input</font>, the vector x
- incx, <font color=#008000> input</font>, the increment of the vector x
- y, <font color=#FF0000> output</font>, the vector y
- incy, <font color=#008000> input</font>, the increment of the vector y

**返回值：**

1. TBLAS_STATUS_SUCCESS 成功
2. TBLAS_STATUS_BAD_PARAM 参数错误
3. TBLAS_STATUS_NOT_SUPPORTED 目前不支持
4. TBLAS_STATUS_NOT_INITIALIZED handle 未初始化


## tblas?copy
```C
tblasStatus_t TBLASWINAPI tblasScopy(
    tblasHandle_t handle, 
    int n, 
    const float *x,
    int incx, 
    float *y, 
    int incy);
tblasStatus_t TBLASWINAPI tblasHcopy(
    tblasHandle_t handle,
    int n, 
    const half *x,
    int incx, 
    half *y, 
    int incy);
```

**功能描述**：

copies the vector x into the vector y

**支持的数据类型**
- float
- half

**参数描述：**
- handle, <font color=#008000> input</font>, handle to the TecoBLAS library
- n, <font color=#008000> input</font>, the length of the vector x
- x, <font color=#008000> input</font>, the vector x
- incx, <font color=#008000> input</font>, the increment of the vector x
- y, <font color=#FF0000> output</font>, the vector y
- incy, <font color=#008000> input</font>, the increment of the vector y

**返回值：**

1. TBLAS_STATUS_SUCCESS 成功
2. TBLAS_STATUS_BAD_PARAM 参数错误
3. TBLAS_STATUS_NOT_SUPPORTEDED 目前不支持
4. TBLAS_STATUS_NOT_INITIALIZED handle 未初始化

**其他限制条件：**
incx，incy支持正整数

## tblas?dot

```C
tblasStatus_t TBLASWINAPI tblasSdot(
    tblasHandle_t handle, 
    int n, 
    const void *x,
    int incx, 
    const void *y, 
    int incy,
    void *result);
tblasStatus_t TBLASWINAPI tblasHdot(
    tblasHandle_t handle, 
    int n, 
    const void *x,
    int incx,
    const void *y,
    int incy,
    void *result);
```

**功能描述**：

computes the dot product of vectors x and y

**支持的数据类型**
- float
- half

**参数描述：**
- handle, <font color=#008000> input</font>, handle to the TecoBLAS library
- n, <font color=#008000> input</font>, the length of the vector x
- x, <font color=#008000> input</font>, the vector x, refers to float(in tblasSdot)/half(in tblasHdot)
- incx, <font color=#008000> input</font>, the increment of the vector x
- y, <font color=#008000> input</font>, the vector y, refers to float(in tblasSdot)/half(in tblasHdot)
- incy, <font color=#008000> input</font>, the increment of the vector y
- result, <font color=#FF0000> output</font>, the result of the dot product, refers to float(in tblasSdot and tblasHdot)

**返回值：**

1. TBLAS_STATUS_SUCCESS 成功
2. TBLAS_STATUS_BAD_PARAM 参数错误
3. TBLAS_STATUS_NOT_SUPPORTEDED 目前不支持
4. TBLAS_STATUS_NOT_INITIALIZED handle 未初始化

**其他限制条件：**
incx，incy支持正整数

## tblas?nrm2

```C
tblasStatus_t TBLASWINAPI tblasSnrm2(
    tblasHandle_t handle, 
    int n, 
    const float *x,
    int incx, 
    float *result);
tblasStatus_t TBLASWINAPI tblasHnrm2(
    tblasHandle_t handle, 
    int n, 
    const half *x,
    int incx, 
    float *result);
```

**功能描述**：

computes the Euclidean norm of the vector x

**支持的数据类型**
- float
- half

**参数描述：**
- handle, <font color=#008000> input</font>, handle to the TecoBLAS library
- n, <font color=#008000> input</font>, the length of the vector x
- x, <font color=#008000> input</font>, the vector x
- incx, <font color=#008000> input</font>, the increment of the vector x
- result, <font color=#FF0000> output</font>, the result norm, float for better precision

**返回值：**

1. TBLAS_STATUS_SUCCESS 成功
2. TBLAS_STATUS_BAD_PARAM 参数错误
3. TBLAS_STATUS_NOT_SUPPORTEDED 目前不支持
4. TBLAS_STATUS_NOT_INITIALIZED handle 未初始化


## tblas?scal

```C
tblasStatus_t TBLASWINAPI tblasSscal(
    tblasHandle_t handle, 
    int n, 
    float alpha,
    float *x, 
    int incx);
tblasStatus_t TBLASWINAPI tblasHscal(
    tblasHandle_t handle, 
    int n, 
    half alpha,
    half *x, int incx);
```
**功能描述**：

scales the vector x by the scalar alpha and overwrites it with the result

**支持的数据类型**
- float
- half

**参数描述：**

-   handle, <font color=#008000> input</font>, handle to the TecoBLAS library
-   n, <font color=#008000> input</font>, the length of the vector x
-   alpha, <font color=#008000> input</font>, the scalar alpha
-   x, <font color=#FF0000> output</font>, the vector x
-   incx, <font color=#008000> input</font>, the increment of the vector x

**返回值：**

1. TBLAS_STATUS_SUCCESS 成功
2. TBLAS_STATUS_BAD_PARAM 参数错误
3. TBLAS_STATUS_NOT_SUPPORTED 目前不支持
4. TBLAS_STATUS_NOT_INITIALIZED handle 未初始化

## tblas?swap

```C
tblasStatus_t TBLASWINAPI tblasSswap(
    tblasHandle_t handle, 
    int n, 
    float *x, 
    int incx,
    float *y, 
    int incy);
tblasStatus_t TBLASWINAPI tblasHswap(
    tblasHandle_t handle, 
    int n, 
    half *x, 
    int incx,
    half *y, 
    int incy);
```
**功能描述**：

interchanges the elements of vector x and y

**支持的数据类型**
- float
- half

**参数描述：**

-   handle, <font color=#008000> input</font>, handle to the TecoBLAS library
-   n, <font color=#008000> input</font>, the length of the vector x
-   x, input and output, the vector x
-   incx, <font color=#008000> input</font>, the increment of the vector x
-   y, input and output, the vector y
-   incy, <font color=#008000> input</font>, the increment of the vector y

**返回值：**

1. TBLAS_STATUS_SUCCESS 成功
2. TBLAS_STATUS_BAD_PARAM 参数错误
3. TBLAS_STATUS_NOT_SUPPORTED 目前不支持
4. TBLAS_STATUS_NOT_INITIALIZED handle 未初始化

**其他限制条件：**
incx，incy支持正整数

## tblasI?rot

```C
tblasStatus_t TBLASWINAPI tblasSrot(
    tblasHandle_t handle, 
    int n, 
    float *x, 
    int incx,
    float *y, 
    int incy, 
    float c, 
    float s);
tblasStatus_t TBLASWINAPI tblasHrot(
    tblasHandle_t handle, 
    int n, 
    half *x, 
    int incx,
    half *y, 
    int incy, 
    float c, 
    float s);
```
**功能描述**：

applies Givens rotation matrix, **currently not implemented**

**支持的数据类型**
- float
- half

**返回值：**

1. TBLAS_STATUS_NOT_SUPPORTED 目前不支持

## tblasI?rotg

```C
tblasStatus_t TBLASWINAPI tblasSrotg(
    tblasHandle_t handle, 
    float *a, 
    float *b,
    float *c, 
    float *s);
tblasStatus_t TBLASWINAPI tblasHrotg(
    tblasHandle_t handle, 
    half *a, 
    half *b, 
    half *c,
    half *s);
```
**功能描述**：

constructs the Givens rotation matrix, **currently not implemented**

**支持的数据类型**
- float
- half

**返回值：**

1. TBLAS_STATUS_NOT_SUPPORTED 目前不支持

## tblasI?rotm

```C
tblasStatus_t TBLASWINAPI tblasSrotm(
    tblasHandle_t handle, 
    int n, 
    float *x, 
    int incx,
    float *y, 
    int incy, 
    const float *param);
tblasStatus_t TBLASWINAPI tblasHrotm(
    tblasHandle_t handle, 
    int n, 
    half *x, 
    int incx,
    half *y, 
    int incy, 
    const half *param);
```
**功能描述**：

applies the modified Givens transformation, **currently not implemented**

**支持的数据类型**
- float
- half

**返回值：**

1. TBLAS_STATUS_NOT_SUPPORTED 目前不支持

## tblasI?rotmg

```C
tblasStatus_t TBLASWINAPI tblasSrotmg(
    tblasHandle_t handle, 
    float *d1, 
    float *d2,
    float *x1, 
    float *x2, 
    float *param);
tblasStatus_t TBLASWINAPI tblasHrotmg(
    tblasHandle_t handle, 
    half *d1, 
    half *d2,
    half *x1, 
    half *x2, 
    half *param);
```
**功能描述**：

constructs the modified Givens transformation, **currently not implemented**

**支持的数据类型**
- float
- half

**返回值：**

1. TBLAS_STATUS_NOT_SUPPORTED 目前不支持


# Level-2 API 列表

## tblas?gemv

```C
tblasStatus_t TBLASWINAPI tblasSgemv(
    tblasHandle_t handle, 
    tblasOperation_t trans,
    int m, 
    int n, 
    float alpha, 
    const void *A,
    int lda, 
    const void *x, 
    int incx, 
    float beta,
    void *y, 
    int incy);
tblasStatus_t TBLASWINAPI tblasHgemv(
    tblasHandle_t handle, 
    tblasOperation_t trans,
    int m, 
    int n, 
    float alpha, 
    const void *A,
    int lda, 
    const void *x, 
    int incx, 
    float beta,
    void *y, 
    int incy); 
```
**功能描述**：

performs the matrix-vector multiplication

**支持的数据类型**
- float
- half

**参数描述：**

-   handle, <font color=#008000> input</font>, handle to the TecoBLAS library
-   trans, <font color=#008000> input</font>, operation op(A)
-   m, <font color=#008000> input</font>, number of rows of matrix A.
-   n, <font color=#008000> input</font>, number of columns of matrix A.
-   alpha, <font color=#008000> input</font>, scalar used for multiplication.
-   A, <font color=#008000> input</font>, array of dimension lda x n with lda >= max(1,m), refers to float(in tblasSgemv)/half(in tblasHgemv).
-   lda, <font color=#008000> input</font>, leading dimension of two-dimensional array used to store matrix A. lda must be at least max(1,m).
-   x, <font color=#008000> input</font>, vector at least n*incx elements if transa==TBLAS_OP_N and at least m*incx elements otherwise, refers to float(in tblasSgemv)/half(in tblasHgemv).
-   incx, <font color=#008000> input</font>, increment of vector x.
-   beta, <font color=#008000> input</font>, scalar used for multiplication.
-   y, <font color=#FF0000> output</font>, vector at least m*incy elements if transa==TBLAS_OP_N and at least n*incy elements otherwise, refers to float(in tblasSgemv)/half(in tblasHgemv).
-   incy, <font color=#008000> input</font>, increment of vector y.

**返回值：**

1. TBLAS_STATUS_SUCCESS 成功
2. TBLAS_STATUS_BAD_PARAM 参数错误
3. TBLAS_STATUS_BAD_LD 参数ld？或者tran?错误
4. TBLAS_STATUS_NOT_INITIALIZED handle 未初始化
5. TBLAS_STATUS_NOT_SUPPORTED 目前不支持

**其他限制条件：**
incx，incy支持正整数
不支持转置

# Level-3 API 列表

## tblas?Tril

```C
tblasStatus_t TBLASWINAPI tblasSTril(
    tblasHandle_t handle, 
    int M, 
    int N, 
    int K, 
    int diagonal,
    void *d_x_f32, 
    void *result);
tblasStatus_t TBLASWINAPI tblasHTril(
    tblasHandle_t handle, 
    int M, 
    int N, 
    int K, 
    int diagonal,
    void *d_x_f16, 
    void *result);
```
**功能描述**：

返回输入矩阵的下三角部分，其余部分被设为0。 矩形的下三角部分被定义为对角线上和下方的元素。

**支持的数据类型**
- float
- half

**参数描述：**

-   handle, <font color=#008000> input</font>, handle to the TecoBLAS library
-   M, <font color=#008000> input</font>, number of rows of matrix op(d_x_f?) and result
-   N, <font color=#008000> input</font>, number of columns of matrix op(d_x_f?) and result
-   K, <font color=#008000> input</font>, number of matrix op(d_x_f?) and result
-   diagonal, <font color=#008000> input</font>, If diagonal=0, it indicates the main diagonal; If the diagonal is a positive number, it indicates the diagonal above the main diagonal; If the diagonal is negative, it indicates the diagonal below the main diagonal
-   d_x_f?, <font color=#008000> input</font>, array of dimension M * N.
-   result, <font color=#FF0000> output</font>, array of dimension M * N.

**返回值：**

1. TBLAS_STATUS_SUCCESS 成功
2. TBLAS_STATUS_BAD_PARAM 参数错误
3. TBLAS_STATUS_NOT_SUPPORTED 目前不支持
4. TBLAS_STATUS_NOT_INITIALIZED handle 未初始化

## tblas?Triu

```C
tblasStatus_t TBLASWINAPI tblasSTriu(
    tblasHandle_t handle, 
    int M, 
    int N, 
    int K, 
    int diagonal,
    void *d_x_f32, 
    void *result);
tblasStatus_t TBLASWINAPI tblasHTriu(
    tblasHandle_t handle, 
    int M, 
    int N, 
    int K, 
    int diagonal,
    void *d_x_f16, 
    void *result);
```
**功能描述**：

返回输入矩阵的上三角部分，其余部分被设为0。 矩形的上三角部分被定义为对角线上和上方的元素。

**支持的数据类型**
- float
- half

**参数描述：**

-   handle, <font color=#008000> input</font>, handle to the TecoBLAS library
-   M, <font color=#008000> input</font>, number of rows of matrix op(d_x_f?) and result
-   N, <font color=#008000> input</font>, number of columns of matrix op(d_x_f?) and result
-   K, <font color=#008000> input</font>, number of matrix op(d_x_f?) and result
-   diagonal, <font color=#008000> input</font>, If diagonal=0, it indicates the main diagonal; If the diagonal is a positive number, it indicates the diagonal above the main diagonal; If the diagonal is negative, it indicates the diagonal below the main diagonal
-   d_x_f?, <font color=#008000> input</font>, array of dimension M * N.
-   result, <font color=#FF0000> output</font>, array of dimension M * N.

**返回值：**

1. TBLAS_STATUS_SUCCESS 成功
2. TBLAS_STATUS_BAD_PARAM 参数错误
3. TBLAS_STATUS_NOT_SUPPORTED 目前不支持
4. TBLAS_STATUS_NOT_INITIALIZED handle 未初始化

## tblas?gemm

```C
tblasStatus_t TBLASWINAPI tblasSgemm(
    tblasHandle_t handle, 
    tblasOperation_t transa,
    tblasOperation_t transb, 
    int m, 
    int n, 
    int k,
    float alpha, 
    const void *A, 
    int lda,
    const void *B, 
    int ldb, 
    float beta, 
    void *C,
    int ldc);
tblasStatus_t TBLASWINAPI tblasHgemm(
    tblasHandle_t handle, 
    tblasOperation_t transa,
    tblasOperation_t transb, 
    int m, 
    int n, 
    int k,
    float alpha, 
    const void *A, 
    int lda,
    const void *B, 
    int ldb, 
    float beta, 
    void *C,
    int ldc);
```
**功能描述**：

performs the matrix-matrix multiplication

**支持的数据类型**
- float
- half

**参数描述：**

-   handle, <font color=#008000> input</font>, handle to the TecoBLAS library
-   transa, <font color=#008000> input</font>, operation op(A)
-   transb, <font color=#008000> input</font>, operation op(B)
-   m, <font color=#008000> input</font>, number of rows of matrix op(A) and C.
-   n, <font color=#008000> input</font>, number of columns of matrix op(B) and C.
-   k, <font color=#008000> input</font>, number of columns of matrix op(A) and rows of matrix op(B).
-   alpha, <font color=#008000> input</font>, scalar used for multiplication.
-   A, <font color=#008000> input</font>, array of dimension lda * k with lda >= max(1,m), refers to float(in tblasSgemm)/half(in tblasHgemm).
-   lda, <font color=#008000> input</font>, leading dimension of two-dimensional array used to store the matrix A
-   B, <font color=#008000> input</font>, array of dimension ldb * n with ldb >= max(1,k), refers to float(in tblasSgemm)/half(in tblasHgemm).
-   ldb, <font color=#008000> input</font>, leading dimension of two-dimensional array used to store the matrix B
-   beta, <font color=#008000> input</font>, scalar used for multiplication.
-   C, <font color=#FF0000> output</font>, array of dimension ldc * n with ldc >= max(1,m), refers to float(in tblasSgemm)/half(in tblasHgemm).
-   ldc, <font color=#008000> input</font>, leading dimension of two-dimensional array used to store the matrix C

**返回值：**

1. TBLAS_STATUS_SUCCESS 成功
2. TBLAS_STATUS_BAD_PARAM 参数错误
3. TBLAS_STATUS_BAD_LD 参数ld？或者tran?错误
4. TBLAS_STATUS_NOT_INITIALIZED handle 未初始化
5. TBLAS_STATUS_NOT_SUPPORTED 目前不支持

**其他限制条件：**
不支持转置

## tblas?gemmBatched

```C
tblasStatus_t TBLASWINAPI tblasSgemmBatched(
    tblasHandle_t handle, 
    tblasOperation_t transa, 
    tblasOperation_t transb, 
    int m,
    int n, 
    int k, 
    float alpha, 
    const void *Aarray[], 
    int lda, 
    const void *Barray[],
    int ldb, 
    float beta, 
    void *Carray[], 
    int ldc, 
    int batchCount);
tblasStatus_t TBLASWINAPI tblasHgemmBatched(
    tblasHandle_t handle, 
    tblasOperation_t transa, 
    tblasOperation_t transb, 
    int m,
    int n, 
    int k, 
    float alpha, 
    const void *Aarray[], 
    int lda, 
    const void *Barray[], 
    int ldb,
    float beta, 
    void *Carray[], 
    int ldc, 
    int batchCount);
```
**功能描述**：

performs the matrix-matrix multiplication of a batch of matrices.

**参数描述：**

-   handle, <font color=#008000> input</font>, handle to the TecoBLAS library
-   transa, <font color=#008000> input</font>, operation op(Aarray[i])
-   transb, <font color=#008000> input</font>, operation op(Barray[i])
-   m, <font color=#008000> input</font>, number of rows of matrix op(Aarray[i]) and C.
-   n, <font color=#008000> input</font>, number of columns of matrix op(Barray[i]) and C.
-   k, <font color=#008000> input</font>, number of columns of matrix op(Aarray[i]) and rows of matrix op(Barray[i]).
-   alpha, <font color=#008000> input</font>, scalar used for multiplication.
-   Aarray, <font color=#008000> input</font>, array of dimension lda * k with lda >= max(1,m), refers to float(in tblasSgemmBatched)/half(in tblasHgemmBatched).
-   lda, <font color=#008000> input</font>, leading dimension of two-dimensional array used to store the matrix Aarray[i]
-   Barray, <font color=#008000> input</font>, array of dimension ldb * n with ldb >= max(1,k), refers to float(in tblasSgemmBatched)/half(in tblasHgemmBatched).
-   ldb, <font color=#008000> input</font>, leading dimension of two-dimensional array used to store the matrix Barray[i]
-   beta, <font color=#008000> input</font>, scalar used for multiplication.
-   Carray, <font color=#FF0000> output</font>, array of dimension ldc * n with ldc >= max(1,m), refers to float(in tblasSgemmBatched)/half(in tblasHgemmBatched).
-   ldc, <font color=#008000> input</font>, leading dimension of two-dimensional array used to store the matrix Carray[i]
-   batchCount, <font color=#008000> input</font>, number of matrices in the batch

**返回值：**

1. TBLAS_STATUS_SUCCESS 成功
2. TBLAS_STATUS_BAD_PARAM 参数错误
3. TBLAS_STATUS_BAD_LD 参数ld？或者tran?错误
4. TBLAS_STATUS_NOT_INITIALIZED handle 未初始化
5. TBLAS_STATUS_NOT_SUPPORTED 目前不支持

**其他限制条件：**
不支持转置

## tblas?gemmStridedBatched

```C
tblasStatus_t TBLASWINAPI tblasSgemmStridedBatched(
    tblasHandle_t handle, 
    tblasOperation_t transa, 
    tblasOperation_t transb, 
    int m,
    int n, 
    int k, 
    float alpha, 
    const float *A, 
    int lda, 
    long long int strideA,
    const float *B, 
    int ldb, 
    long long int strideB, 
    float beta, 
    float *C, 
    int ldc,
    long long int strideC, 
    int batchCount);
tblasStatus_t TBLASWINAPI tblasHgemmStridedBatched(
    tblasHandle_t handle, 
    tblasOperation_t transa, 
    tblasOperation_t transb, 
    int m,
    int n, 
    int k, 
    half alpha, 
    const half *A, 
    int lda, 
    long long int strideA,
    const half *B, 
    int ldb, 
    long long int strideB, 
    half beta, 
    half *C, 
    int ldc,
    long long int strideC, 
    int batchCount);
```
**功能描述**：

performs the matrix-matrix multiplication of a batch of matrices.

**参数描述：**

-   handle, <font color=#008000> input</font>, the handle to the TecoBLAS library
-   transa, <font color=#008000> input</font>, operation op(A[i])
-   transb, <font color=#008000> input</font>, operation op(B[i])
-   m, <font color=#008000> input</font>, number of rows of matrix op(A[i]) and C.
-   n, <font color=#008000> input</font>, number of columns of matrix op(B[i]) and C.
-   k, <font color=#008000> input</font>, number of columns of matrix op(A[i]) and rows of matrix op(B[i]).
-   alpha, <font color=#008000> input</font>, scalar used for multiplication.
-   A, <font color=#008000> input</font>, array of dimension lda * k with lda >= max(1,m).
-   lda, <font color=#008000> input</font>, leading dimension of two-dimensional array used to store the matrix
-   strideA, <font color=#008000> input</font>, Value of type long long int that gives the offset in number of elements between A[i] and A[i+1]
-   B, <font color=#008000> input</font>, array of dimension ldb * n with ldb >= max(1,k).
-   ldb, <font color=#008000> input</font>, leading dimension of two-dimensional array used to store the matrix
-   strideB, <font color=#008000> input</font>, Value of type long long int that gives the offset in number of elements between B[i] and B[i+1]
-   beta, <font color=#008000> input</font>, scalar used for multiplication.
-   C, <font color=#FF0000> output</font>, array of dimension ldc * n with ldc >= max(1,m).
-   ldc, <font color=#008000> input</font>, leading dimension of two-dimensional array used to store the matrix
-   strideC, <font color=#008000> input</font>,   Value of type long long int that gives the offset in number of elements between C[i] and C[i+1]
-   batchCount, <font color=#008000> input</font>, number of matrices in the batch

**返回值：**

1. TBLAS_STATUS_SUCCESS 成功
2. TBLAS_STATUS_BAD_PARAM 参数错误
3. TBLAS_STATUS_BAD_LD 参数ld？或者tran?错误
4. TBLAS_STATUS_NOT_INITIALIZED handle 未初始化
5. TBLAS_STATUS_NOT_SUPPORTED 目前不支持

**其他限制条件：**
暂不支持此算子


## tblas?addBatchGemm

``` c

tblasStatus_t TBLASWINAPI tblasSaddBatchGemm(
                                            tblasHandle_t handle,
                                            tblasOperation_t transA,
                                            tblasOperation_t transB,
                                            int m,
                                            int n,
                                            int k,
                                            float alpha,
                                            const void *ABatched[],
                                            int lda,
                                            const void *BBatched[],
                                            int ldb,
                                            float beta,
                                            void *C,
                                            int ldc,
                                            int batchCount
                                            );


tblasStatus_t TBLASWINAPI tblasHaddBatchGemm(
                                            tblasHandle_t handle,
                                            tblasOperation_t transA,
                                            tblasOperation_t transB,
                                            int m,
                                            int n,
                                            int k,
                                            float alpha,
                                            const void *ABatched[],
                                            int lda,
                                            const void *BBatched[],
                                            int ldb,
                                            float beta,
                                            void *C,
                                            int ldc,
                                            int batchCount
                                            );

```

**功能描述**：

given batched matrixes `ABatch`, `BBatch` and single matrixes `C`  along with real number `alpha, beta`, calculate result `C = beta*C + alpha*sum(Abatch@Bbatch)`, in which `@` means gemm. Note that C is both input and output.

**参数描述：**

-   handle, <font color=#008000> input</font>, the handle to the TecoBLAS library
-   transa, <font color=#008000> input</font>, operation op(ABatched[i])
-   transb, <font color=#008000> input</font>, operation op(BBatched[i])
-   m, <font color=#008000> input</font>, number of rows of matrix op(ABatched[i]) and C.
-   n, <font color=#008000> input</font>, number of columns of matrix op(BBatched[i]) and C.
-   k, <font color=#008000> input</font>, number of columns of matrix op(ABatched[i]) and rows of matrix op(BBatched[i]).
-   alpha, <font color=#008000> input</font>, scalar used for multiplication.
-   ABatched, <font color=#008000> input</font>, array of dimension lda * k with lda >= max(1,m).
-   lda, <font color=#008000> input</font>, leading dimension of two-dimensional array used to store the matrix
-   strideA, <font color=#008000> input</font>, Value of type long long int that gives the offset in number of elements between ABatched[i] and ABatched[i+1]
-   BBatched, <font color=#008000> input</font>, array of dimension ldb * n with ldb >= max(1,k).
-   ldb, <font color=#008000> input</font>, leading dimension of two-dimensional array used to store the matrix
-   strideB, <font color=#008000> input</font>, Value of type long long int that gives the offset in number of elements between BBatched[i] and BBatched[i+1]
-   beta, <font color=#008000> input</font>, scalar used for multiplication.
-   C, <font color=#FF0000> output</font>, array of dimension ldc * n with ldc >= max(1,m).
-   ldc, <font color=#008000> input</font>, leading dimension of two-dimensional array used to store the matrix
-   strideC, <font color=#008000> input</font>,   Value of type long long int that gives the offset in number of elements between C[i] and C[i+1]
-   batchCount, <font color=#008000> input</font>, number of matrices in the batch

**返回值：**

1. TBLAS_STATUS_SUCCESS 成功
2. TBLAS_STATUS_BAD_PARAM 参数错误
3. TBLAS_STATUS_BAD_LD 参数ld？或者tran?错误
4. TBLAS_STATUS_NOT_INITIALIZED handle 未初始化
5. TBLAS_STATUS_NOT_SUPPORTED 目前不支持

**其他限制条件：**
无
