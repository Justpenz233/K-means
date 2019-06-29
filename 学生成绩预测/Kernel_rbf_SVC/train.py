import xlrd
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib

def Kernel_SVM():
    return SVC(kernel='rbf')

def Narray(t_list, t_type=np.float):
    return np.asarray(t_list, t_type)

# read from file


def ReadFromFile(FeatureFileName, LabelFileName):
    FeatureList = []
    LabelList = []
    sheetA = xlrd.open_workbook(filename=FeatureFileName).sheet_by_index(0)
    sheetB = xlrd.open_workbook(filename=LabelFileName).sheet_by_index(0)
    rowNumber = sheetA.nrows
    for i in range(rowNumber):
        if i == 0:
            continue
        Tlist = sheetA.row_values(i)[1:]
        if Tlist[0] == u'女':
            Tlist[0] = 0.0
        else:
            Tlist[0] = 1.0
        FeatureList.append([1 if i == '' else float(i) for i in Tlist])
        Tlist = sheetB.row_values(i)[1:]
        for index, j in enumerate(Tlist):
            if j == u'不及格':Tlist[index] = 0
            else:Tlist[index] = 1
        LabelList.append([float(i) for i in Tlist])
    FeatureList = Narray(FeatureList)
    LabelList = Narray(LabelList)
    return FeatureList, LabelList


def train(project):
    print('Start to train %d/3..' % project)
    train_x_list, train_y_list = ReadFromFile(
        'train_feature.xlsx', 'train_label.xlsx')
    # 对非线性可分的数据集，简单的 SVM 算法就不再适用。
    # 一种改进的 SVM 叫做 Kernel SVM 可以用来解决非线性可分数据的分类问题。
    # 构造一个高斯曲率支持向量机并训练
    # 必须指定类 SVC 的核参数额值为 “rbf”
    svclassifier = Kernel_SVM()

    # 一个维度一个维度的进行模型训练
    ty_list = [i[project] for i in train_y_list]
    ty_list = Narray(ty_list)
    
    # 训练模型 时间比较久
    svclassifier.fit(train_x_list, ty_list)
    
    # 保存训练过的模型
    File_NAME = 'module' + str(project) + '.m'
    joblib.dump(svclassifier, File_NAME)  # 将模型储存下来


if __name__ == '__main__':
    for i in range(0, 4):
        train(i)
