'''
一个解集即一个个体

支持操作
凭借01矩阵产生初始解 over
#一个解集 to 一个染色体
#一个染色体 to 一个解集
两个个体 诞生 新个体（杂交） p_a
同时有一定几率交叉互换 p_b
同时有一个几率基因突变 p_c
'''
import matplotlib.pyplot as plt
import math
import random
import numpy as np
import global_var as gl
import Solution as SLT

def get_init(mp, grid_len,is_first):
    if not is_first:
        return
    gl.mmp = mp
    gl.row = mp.shape[0]
    gl.col = mp[0].shape[0]
    print(gl.row)
    print(gl.col)
    gl.radius /= grid_len
    gl.max_flight_radius /= grid_len
    for r in range(gl.row):
        for c in range (gl.col):
            if (mp[r][c] == 1):
                gl.px_list.append(c)
                gl.py_list.append(r)
    gl.p_valid_cnt = len(gl.px_list)

    gl.chro_point_len = int(math.ceil(math.log2(gl.p_valid_cnt)))
    gl.chro_total_len = gl.chro_point_len * gl.point_cnt;


def init_random_one_individual():
    '''
    初始随机化一个个体的染色体chromosome
    :return:
    '''
    chromosome = []
    for i in range(gl.chro_total_len):
        chromosome.append(random.randint(0,1))
    return chromosome


def init_random_population(want_pop_size):
    population = list()
    for i in range(want_pop_size):
        while True:
            individual = init_random_one_individual()
            X,Y = individual_to_solution(individual)
            if True == SLT.is_valid(X,Y,gl.mmp):
                break
        population.append(individual)
    return population


def individual_to_solution(individual):
    '''
    :return:
    '''
    X = list()
    Y = list()
    t = 0
    for i in range(gl.chro_total_len):
        t = t*2+individual[i]
        if ((i+1)%gl.chro_point_len == 0):
            t %= gl.p_valid_cnt
            x,y = gl.px_list[t],gl.py_list[t]
            X.append(x)
            Y.append(y)
            # reset 0
            # t = 0
    return X,Y


def crossover(pop):
    '''
    相邻个体以一定概率杂交
    :param pplt:这一代的种群
    :param [out]
    :return:
    '''
    num = len(pop)
    for i in range(num):
        male = 0
        # 0-1的随机数落于[0,cross_p)即杂交概率为cross_p
        if (random.random() < gl.cross_p):
            # 与相邻的对象杂交
            male = (i + 1) % num
            # 随机杂交点数目及杂交点分布
            cnt, cross = random_cross_solution(gl.min_cross_point_cnt, gl.max_cross_point_cnt, gl.chro_total_len)
            produce_new_list(pop[i], pop[male], cnt, cross)


def get_random_list_without_repetition(n, m):
    '''
    n个数里面随机选m个不重复的数
    :param n:
    :param m:
    :return:
    '''
    i = 0
    ans = []
    while i < n:
        if (random.randint(0, n - i - 1) < m):
            ans.append(i)
        i += 1
    return ans


def random_cross_solution(a, b, sz):
    # 随机产生要杂交断裂的断点数目
    cnt = random.randint(a, b)
    l = get_random_list_without_repetition(n=sz, m=cnt)
    # 把最后的点设置为断点
    cnt += 1
    l.append(sz)
    return cnt, l


def produce_new_list(dad, mom, cnt, cross):
    l = 0
    for i,r in enumerate(cross):
        if i%2 == 1:
            for j in range(l,r):
                t = dad[j]
                dad[j]=mom[j]
                mom[j]=t

def mutation(pop):
    '''
    基因突变
    每个新生的个体，以一定概率基因突变（一个位产生变化）
    :return: 
    '''
    num = len(pop)
    ans = []
    for i in range(num):
        if (random.random() < gl.mutation_p):
            j = random.randint(0, gl.chro_total_len)
            pop[i][j] ^= 1
        X,Y = individual_to_solution(individual=pop[i])
        #if True == SLT.is_valid(X=X, Y=Y, mmp=mmp):
        #    ans.append()