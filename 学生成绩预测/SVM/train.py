import xlrd
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib

Train_DataSet_File = 'train_feature.xlsx'
Train_LabelSet_File = 'train_label.xlsx'


# 将list型打包成np array型
def MakeArray(Tlist, tType=np.float):
    return np.asarray(Tlist, tType)

# 从Excel中读取数据并存储
def ReadExcel(file_a, file_b):
    FeatureList = []
    LabelList = []
    xd = xlrd.open_workbook(filename=file_a)
    sheetA = xd.sheet_by_index(0)# 读取feature文件
    xd = xlrd.open_workbook(filename=file_b)
    sheetB = xd.sheet_by_index(0)# 读取label文件
    rowNumber = sheetA.nrows

    for i in range(rowNumber):
        if i == 0: continue
        Tlist = sheetA.row_values(i)[1:]
        # 将男女映射到 0 / 1
        if Tlist[0] == u'女': Tlist[0] = 0.0
        else: Tlist[0] = 1.0 
        # 数据中有空白格，需要单独处理下
        FeatureList.append([1 if i == '' else float(i) for i in Tlist]) 
        Tlist = sheetB.row_values(i)[1:]
        # 将不及格与及格映射到 0 / 1
        for index, j in enumerate(Tlist):
            if j == u'不及格': Tlist[index] = 0
            else: Tlist[index] = 1 
        LabelList.append([float(i) for i in Tlist])
    FeatureList = MakeArray(FeatureList)
    LabelList = MakeArray(LabelList)
    # 返回打包好的Feature和Label
    return FeatureList, LabelList


# 训练过程
def train(project):
    print('Start to train %d/3..'% project)
    train_x_list, train_y_list = ReadExcel(
        Train_DataSet_File, Train_LabelSet_File)
    #训练一个线性SVM
    svclassifier = SVC(kernel='linear')
    ty_list = [i[project] for i in train_y_list]
    ty_list = MakeArray(ty_list)

    svclassifier.fit(train_x_list, ty_list)

    save_path_name = str(project) + '.m'
    joblib.dump(svclassifier, save_path_name)


if __name__ == '__main__':
    for i in range(0, 4):
        train(i)
