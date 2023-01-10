import numpy as np

# 生成混淆矩阵，n_cl是类别数，gt是H×W的真实值，pred是H×W的预测值
def hist_info(n_cl, pred, gt):
    assert (pred.shape == gt.shape)  # gt的形状和pred的形状必须相等
    k = (gt >= 0) & (gt < n_cl)  # 生成一个H×W的mask，里面类别属于0~n_cl的为True，否则为False
    labeled = np.sum(k)  # 统计mask中元素为True的数量，即标签类别是0~n_cl的元素个数
    correct = np.sum((pred[k] == gt[k]))  # 统计预测正确的数量

    # 返回混淆矩阵、统计元素的数量和预测正确元素的数量。
    return np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int),
                       minlength=n_cl ** 2).reshape(n_cl, n_cl),labeled,correct


def compute_miou(hist,labeled,correct):
    np.seterr(divide='ignore', invalid='ignore')
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))  # 交并比 = 对角线元素 / 所有元素和 -对角线元素

    mean_IU = np.nanmean(iu)  # numpy.nanmean()函数可用于计算忽略NaN值的数组平均值,经过这一步就算得了MIOU

    mean_IU_no_back = np.nanmean(iu[1:])  # 除去背景的MIOU（0类别代表背景）

    mpa = correct / labeled  # 平均像素准确率 = 预测正确像素个数 / 总像素个数
    return iu, mean_IU, mean_IU_no_back, mpa
