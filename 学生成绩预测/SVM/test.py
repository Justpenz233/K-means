import train
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib

def MakeAnswer(project):
    TestXList, TestyList = train.ReadExcel('test_feature.xlsx', 'test_label.xlsx')
    save_path_name = str(project) + '.m'
    svc = joblib.load(save_path_name)
    pre = svc.predict(TestXList)
    return pre, [i[project] for i in TestyList]

def mk_diff():
    pro_list = ['800/1000米', '50米', '立定跳远', '引体/仰卧']
    sum_total = 0
    sum_correct = 0
    for i in range(0, 4):
        pre, origin = MakeAnswer(i)
        total = 0
        correct = 0
        for index, j in enumerate(pre):
            total += 1
            if j == origin[index]:
                correct += 1
        print('Project ' + pro_list[int(i)] + ' correct rate : ', correct / total)
        sum_total += total
        sum_correct += correct
    print('Total correct rate : ', sum_correct / sum_total)



if __name__ == '__main__':
    mk_diff()
