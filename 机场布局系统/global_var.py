'''

'''

radius = 60.0
row = 0
col = 0
chro_point_len = 0
chro_total_len = 0
point_cnt = 65 #number of Airports
want_init_population_size = 15
max_population_size = 59
cross_p = 0.6
mutation_p = 0.01
min_cross_point_cnt = 3
max_cross_point_cnt = 6
max_flight_radius = 300
iter_count = 10 # iteration  times
px_list = list()
py_list = list()
p_valid_cnt = 0
mmp=None
Max_Point_Num = 35


def set_point_cnt(self,n):
    global point_cnt
    point_cnt = n

def print_for_check():
    print("_____")
    print("radius = ",radius)
    print("row = ", row)
    print("col = ",col)
    print("point_cnt = ", point_cnt)
    print("want_init_population_size = ", want_init_population_size)
    print("max_population_size = ", max_population_size)
    print("cross_p = ", cross_p)
    print("mutation_p = ", mutation_p)
    print("min_cross_point_cnt = ", min_cross_point_cnt)
    print("max_cross_point_cnt = ", max_cross_point_cnt)
    print("max_flight_radius = ", max_flight_radius)
    print("_(-_-)__")