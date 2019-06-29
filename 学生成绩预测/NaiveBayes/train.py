import xlrd
import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.externals import joblib

Train_DataSet_File = 'train_feature.xlsx'
Train_LabelSet_File = 'train_label.xlsx'


# make a numpy ndarray
def mk_array(t_list, t_type=np.float):
    return np.asarray(t_list, t_type)


# read from file
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


# train process
def train(project):
    train_x_list, train_y_list = read_file(Train_DataSet_File, Train_LabelSet_File)

    gnb = GaussianNB()
    ty_list = [i[project] for i in train_y_list]
    ty_list = mk_array(ty_list)

    gnb.fit(train_x_list, ty_list)

    save_path_name = str(project) + '.m'
    joblib.dump(gnb, save_path_name)


if __name__ == '__main__':
    for i in range(0, 4):
        train(i)
