# IMBD 2022 - ProjectB Final Round
###### tags: `Doc`

> 1. The complete workflow is organized in [111011_projectB_report.pdf](https://github.com/JiangJiaWei1103/IMBD-2022-ProjectB-Final/blob/master/111011_projectB_report.pdf).

## Objective
Given processing information (*i.e.,* `n_sg.csv` (伺服資訊，以1000Hz採樣) and `n_spike.csv` (智慧刀把資訊，以2500Hz採樣), where n is the indicator of layers) of target knives, the objective is to predict the **maximal flank wear (刀腹摩耗最大值)** `MaxWear` after each processing stage (layer). There are 3 knives in total (`train1` is the first one with 46 layers of processing, `train2` is the second one with 24 layers and the last `test` with 25). The evaluation metric is *RMSE*.

## How to Run
To obtain the predicting results on different datasets, namely `train1`, `train2`, and `test`, please follow commands written in section **執行方式** of [111011_projectB_report.pdf](https://github.com/JiangJiaWei1103/IMBD-2022-ProjectB-Final/blob/master/111011_projectB_report.pdf).

## Performance
*RMSE=0.052979054*，通過銀獎資格門檻，**但未獲獎**。<br>

## Conclusion
由於主辦單位不便透露競賽細節及排名，故較難找出本次競賽的癥結所在，我們總結競賽失利的可能原因如下:
1. Fail to create **reliable CV scheme**.
2. Fail to generate **robust features**, which may boost the generalizability of base models.
3. Leverage only naive feature selection methods, but abandon advanced ones (*e.g.,* forward selection, RFE).

最終結果明顯overfit在training set跟validation set上，造成unseen data (*e.g.,* `test`) 的performance不如預期。
