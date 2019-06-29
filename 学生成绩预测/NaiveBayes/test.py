import xlrd
import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.externals import joblib

Test_DataSet_File = 'test_feature.xlsx'
Test_LabelSet_File = 'test_label.xlsx'


def mk_array(t_list, t_type=np.float):
    return np.asarray(t_list, t_type)


def read_file(file_a, file_b):
    data_list = []
    label_list = []

    sheet1 = xlrd.open_workbook(filename=file_a).sheet_by_index(0)
    sheet2 = xlrd.open_workbook(filename=file_b).sheet_by_index(0)
    n_row = sheet1.nrows

    for i in range(n_row):
        if i == 0:
            continue
        t_list = sheet1.row_values(i)[1:]
        if t_list[0] == u'女':
            t_list[0] = 0.0
        else:
            t_list[0] = 1.0

        data_list.append([0 if i == '' else float(i) for i in t_list])

        t_list = sheet2.row_values(i)[1:]
        for index, j in enumerate(t_list):
            if j == u'不及格':
                t_list[index] = 0
            else:
                t_list[index] = 1
        label_list.append([float(i) for i in t_list])

    return mk_array(data_list), mk_array(label_list)


def predict(project):
    test_x_list, test_y_list = read_file(Test_DataSet_File, Test_LabelSet_File)
    save_path_name = str(project) + '.m'
    gnb = joblib.load(save_path_name)
    pre = gnb.predict(test_x_list)
    return pre, [i[project] for i in test_y_list]


def mk_diff():
    pro_list = ['800/1000米', '50米', '立定跳远', '引体/仰卧']
    sum_total = 0
    sum_correct = 0
    for i in range(0, 4):
        pre, origin = predict(i)
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
