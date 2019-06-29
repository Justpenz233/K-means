import xlrd
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import classification_report, confusion_matrix
import train

datasetfile = 'test_feature.xlsx'
labelsetfile = 'test_label.xlsx'

def predict(project):
    test_x_list, test_y_list = train.ReadFromFile(datasetfile, labelsetfile)
    save_path_name = 'module' + str(project) + '.m'
    # 读取模型并保存
    svc = joblib.load(save_path_name)
    # 使用模型预测
    pre = svc.predict(test_x_list)
    return pre, [i[project] for i in test_y_list]

def TEST_module():
    # 对四个维度进行预测
    for i in range(0, 4):
        y_pre, y_ori = predict(i)
        print(confusion_matrix(y_ori, y_pre))
        print(classification_report(y_ori, y_pre))


if __name__ == '__main__':
    TEST_module()
