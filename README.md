# 112-2 ADCM Homework 2

## 作業目的

此作業目的是利用 MPI 平行計算出萬個二維座標點中，兩兩座標點所形成的座標對距離前五近為何五對以及該五對對內兩座標點間的確切距離。

## 輸入輸出範例

### 座標點資料範例

可使用亂數隨機生成
```
1.2    1.3
2.4    2.3
4.3    4.3
8.4    8.4
15.1   16.7

```

### 程式結果輸出範例

輸出格式為: 距離  (座標點1, 座標點2)

```
1.562050  (0, 1)
2.758623  (1, 2)
4.313931  (0, 2)
5.798276  (2, 3)
8.556284  (1, 3)
```

## 執行

使用以下指令在裝有 `make` 與 `mpicc` 編譯器的環境編譯該程式
```
make
```

使用以下指令編譯帶有除錯輸出的版本
```
make debug
```

使用以下指令來清除已編譯的檔案 (在未修改檔案的情況下若要重新編譯，需先下此指令)
```
make clean
```

編譯完成後，在裝有 `OpenMPI` 的環境使用該指令來執行此程式
```
mpiexec -np <進程數> 112525005_hw2
```