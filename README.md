K-Means
=======

## 先说结论
* 单线程优化：~18倍
* 2线程优化：~35倍
* 4线程优化：~68倍
* 8线程优化：~130倍
* 12线程优化：~180倍

## 设备信息
* CPU型号：AMD Ryzen 9 5900X
* 核心数：12（关闭超线程）
* CPU频率：锁定100\*47=4700MHz（实际98.7\~99.3\*47=4638.9\~4667.1MHz）
* 内存：32GB@3200MHz
* 运行环境：Debian@5.10.16.3-microsoft-standard-WSL2
* 编译器：gcc version 10.3.0 (Debian 10.3.0-11)

## 优化方法
参见git commit记录，部分摘要：
* 移除`iostream`的flush
* 将`std::vector::assign`替换为`__builtin_memset`
* 将`AssignmentBak`移到循环外
* 将`std::vector`的`operator==`替换为`__builtin_memcmp`
* 将`Assignment[i]`改为临时变量存储
* 将`Point::Distance`标为inline
* 将`Centers`和`PointCount`合并
* 使用SIMD优化（AVX2、FMA）
  + 临时数组增加对齐要求
  + 用intrinsics重写运算逻辑
  + 重写对SIMD友好的`Point`类
* 使用OpenMP优化
  + `#pragma omp parallel for`
  + 手动分配`UpdateCenters`的任务
  + 注意对齐和`Centers`内数据间留空，排除false sharing
* 多写几个版本多跑几次benchmark

经过正确性测试和性能测试后，有下列优化结果：
|#Point |#Cluster|#Iteration|Original |This (#Thread=1)  |This (#Thread=2)          |This(#Thread=4)           |This(#Thread=8)           |This(#Thread=12)          |
|-------|--------|----------|---------|------------------|--------------------------|--------------------------|--------------------------|--------------------------|
|1000000|20      |444       | 26.9359s| 1.52317s (17.68x)|0.777098s (34.66x,  1.96x)|0.391776s (68.75x, 3.888x)|0.200378s (134.4x, 7.601x)| 0.14258s (188.9x, 10.68x)|
|1000000|20      |343       | 20.0322s| 1.18144s (16.96x)|0.590984s ( 33.9x, 1.999x)|0.299734s (66.83x, 3.942x)|0.155514s (128.8x, 7.597x)|0.110057s (  182x, 10.73x)|
|1000000|20      |222       | 13.8133s|0.760443s (18.16x)|0.386173s (35.77x, 1.969x)|0.197917s (69.79x, 3.842x)|0.103292s (133.7x, 7.362x)|0.077298s (178.7x, 9.838x)|
|1000000|15      |182       | 9.68754s|0.511022s (18.96x)|0.264259s (36.66x, 1.934x)|0.138186s (70.11x, 3.698x)|0.070483s (137.4x,  7.25x)|0.053322s (181.7x, 9.584x)|
|1000000|10      |126       | 5.01411s|0.291667s (17.19x)|0.151066s (33.19x, 1.931x)|0.076771s (65.31x, 3.799x)|0.042378s (118.3x, 6.883x)|0.032003s (156.7x, 9.114x)|
|1000000| 5      | 11       |0.267094s|0.020406s (13.09x)|0.011076s (24.11x, 1.842x)|0.006586s (40.55x, 3.098x)|0.005845s ( 45.7x, 3.491x)|0.006187s (43.17x, 3.298x)|
|1000000| 1      |  1       |0.003968s|0.002684s (1.478x)|0.002347s (1.691x, 1.144x)| 0.00243s (1.633x, 1.105x)|0.002427s (1.635x, 1.106x)|0.002958s (1.341x, 0.9074x)|
|100000 |18      |147       |0.806276s|0.047428s (   17x)|0.024328s (33.14x,  1.95x)|0.012607s (63.95x, 3.762x)|0.007058s (114.2x,  6.72x)|0.006914s (116.6x,  6.86x)|
注：
* This (#Thread=1)列括号中的数字是`(OriginalTime/ThisTime)`
* This (#Thread=N)列括号中的数字是`(OriginalTime/ThisTime, Thread1Time/ThisTime)`

## 参考资料
* [Intel(R) Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
* [Intel(R) 64 and IA-32 Architectures Software DeveloperManuals](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
* [OpenMP API Specification](https://www.openmp.org/spec-html/5.0/openmp.html)
* [How to overclock your PC's CPU](https://www.pcworld.com/article/406279/how-to-overclock-your-pcs-cpu.html)
  + 注：用于学习如何固定CPU频率
