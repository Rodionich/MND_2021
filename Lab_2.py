from random import randint
import math
import numpy as np

#Задані дані
N_var = 9
Y_max = (30 - N_var)*10
Y_min = (20 - N_var)*10
X1_min = -20
X1_max = 15
X2_min = -30
X2_max = 45
N = 5

#Значення критерію Романовського за різних довірчих ймовірностей p кількостях дослідів m
p_list = (0.99, 0.98, 0.95, 0.90)
rkr_table = {2: (1.73, 1.72, 1.71, 1.69),
             6: (2.16, 2.13, 2.10, 2.00),
             8: (2.43, 4.37, 2.27, 2.17),
             10: (2.62, 2.54, 2.41, 2.29),
             12: (2.75, 2.66, 2.52, 2.39),
             15: (2.9, 2.8, 2.64, 2.49),
             20: (3.08, 2.96, 2.78, 2.62)}

#Заповнимо матрицю планування для m=5
matrix = [[randint(Y_min, Y_max) for n in range(N)] for k in range(3)]
x_norm = [[-1, 1, -1], [-1, -1, 1]]

#---Перевірка однорідності дисперсії за критерієм Романовського---
#1.Знайдемо середнє значення функції відгуку в рядку:
average_Y1 = sum(matrix[0][j] for j in range(N))/N
average_Y2 = sum(matrix[1][j] for j in range(N))/N
average_Y3 = sum(matrix[2][j] for j in range(N))/N
#2.Знайдемо дисперсії по рядках:
D_Y1 = sum([(j - average_Y1) ** 2 for j in matrix[0]]) / N
D_Y2 = sum([(j - average_Y2) ** 2 for j in matrix[1]]) / N
D_Y3 = sum([(j - average_Y3) ** 2 for j in matrix[2]]) / N
#3.Обчислимо основне відхилення:
main_deviation = math.sqrt((2 * (2 * N - 2)) / (N * (N - 4)))
#4.Обчислимо Fuv:
Fuv_1 = D_Y1 / D_Y2
Fuv_2 = D_Y3 / D_Y1
Fuv_3 = D_Y3 / D_Y2
#4.Обчислимо TETAuv:
TETAuv_1 = ((N - 2) / N) * Fuv_1
TETAuv_2 = ((N - 2) / N) * Fuv_2
TETAuv_3 = ((N - 2) / N) * Fuv_3
#6.Обчислимо Ruv:
Ruv_1 = abs(TETAuv_1 - 1) / main_deviation
Ruv_2 = abs(TETAuv_2 - 1) / main_deviation
Ruv_3 = abs(TETAuv_3 - 1) / main_deviation

#Розрахуємо нормованих коефіцієнтів рівняння регресії.
mx1 = sum(x_norm[0]) / 3
mx2 = sum(x_norm[1]) / 3

my = (average_Y1 + average_Y2 + average_Y3)/3

a1 = sum([i**2 for i in x_norm[0]])/3
a2 = sum (x_norm[0][i] * x_norm[1][i] for i in range(3)) / 3
a3 = sum([i**2 for i in x_norm[1]])/3
a11 = (x_norm[0][0] * average_Y1 + x_norm[0][1] * average_Y2 + x_norm[0][2] * average_Y3) / 3
a22 = (x_norm[1][0] * average_Y1 + x_norm[1][1] * average_Y2 + x_norm[1][2] * average_Y3) / 3

B0 = np.linalg.det([[my, mx1, mx2],[a11, a1, a2], [a22, a2, a3]]) / (np.linalg.det([[1, mx1, mx2], [mx1, a1, a2], [mx2, a2, a3]]))
B1 = np.linalg.det([[1, my, mx2], [mx1, a11, a2], [mx2, a22, a3]]) / (np.linalg.det([[1, mx1, mx2], [mx1, a1, a2],[mx2, a2, a3]]))
B2 = np.linalg.det([[1, mx1, my],[mx1, a1, a11],[mx2, a2, a22]]) / (np.linalg.det([[1, mx1, mx2],[mx1, a1, a2],[mx2, a2, a3]]))

#Проводимо натуралізацію коефіцієнтів:
delta_x1 = math.fabs(X1_max - X1_min) / 2
delta_x2 = math.fabs(X2_max - X2_min) / 2
x10 = (X1_max + X1_min) / 2
x20 = (X2_max + X2_min) / 2
a2_0 = B0 - (B1 * (x10 / delta_x1)) - (B2 * (x20 / delta_x2))
a2_1 = B1 / delta_x1
a2_2 = B2 / delta_x2

#---Перевірка однорідності дисперсії за критерієм Романовського---
def check_homogeneity():
    m = min(rkr_table, key=lambda x: abs(x - N))
    p = 0
    for ruv in (Ruv_1, Ruv_2, Ruv_3):
        if ruv > rkr_table[m][0]:
            print("Дисперсія не є однорідною ! Перезапустіть програму !")
            import sys
            sys.exit(0)
        for rkr in range(len(rkr_table[m])):
            if ruv < rkr_table[m][rkr]:
                p = rkr
    temp = rkr_table[m][p]
    return p_list[p], temp
p, item_table = check_homogeneity()

print("Дані варіанту 209 : Y_max = {}  Y_min = {}  X1_min = {}  X1_max = {}  X2_min = {}  X2_max = {}".format(Y_max, Y_min, X1_min, X1_max, X2_min, X2_max))
print("Матриця планування для m=5")
print("-"*53)
print("| X1  | X2  |  Y1   |  Y2   |  Y3   |  Y4   |  Y5   |")
print("-"*53)
for i in range(3):
    print(f"| {x_norm[0][i]:^4}| {x_norm[1][i]:^4}| {matrix[i][0]:^6}| {matrix[i][1]:^6}| {matrix[i][2]:^6}| {matrix[i][3]:^6}| {matrix[i][4]:^6}|")
print("-"*53)
print("1) Перевірка однорідності дисперсії за критерієм Романовського:")
print("1. Cереднє значення функції відгуку в рядку: Y1 = {}  Y2 = {}  Y3 = {}".format(average_Y1, average_Y2, average_Y3))
print("2. Значення дисперсії по рядках: σ²(Y1) = {}  σ²(Y2) = {}  σ²(Y3) = {}".format("%.2f" % D_Y1, "%.2f" % D_Y2, "%.2f" % D_Y3))
print("3. Основне відхилення σθ: {}".format("%.2f" % main_deviation))
print("4. Обчислюємо Fuv: Fuv_1 = {}  Fuv_2 = {}  Fuv_3 = {}".format("%.2f" % Fuv_1, "%.2f" % Fuv_2, "%.2f" % Fuv_3))
print("5. Обчислюємо θuv: θ_uv1 = {}  θ_uv2 = {}  θ_uv3 = {}".format("%.2f" % TETAuv_1, "%.2f" % TETAuv_2, "%.2f" % TETAuv_3))
print("6. Обчислюємо Ruv: Ruv_1 = {}  Ruv_2 = {}  Ruv_3 = {}".format("%.2f" % Ruv_1, "%.2f" % Ruv_2, "%.2f" % Ruv_3))
print("Ruv1 = {} < Rкр = {}".format("%.2f" % Ruv_1, item_table))
print("Ruv2 = {} < Rкр = {}".format("%.2f" % Ruv_2, item_table))
print("Ruv3 = {} < Rкр = {}".format("%.2f" % Ruv_3, item_table))
print("Однорідність дисперсій підтверджується з ймовірністю p = {} !".format(p))
print("2) Розрахунок нормованих коефіцієнтів рівняння регресії:")
print("mx1 = {}  mx2 = {}  my = {}".format("%.2f" % mx1, "%.2f" % mx2, "%.2f" % my))
print("a1 = {}  a2 = {}  a3 = {}".format("%.2f" % a1, "%.2f" % a2, "%.2f" % a3))
print("a11 = {}  a22 = {}     =>    B0 = {}  B1 = {}  B2 = {}".format("%.2f" % a11, "%.2f" % a22, "%.2f" % B0, "%.2f" % B1, "%.2f" % B2))
print("Нормоване рівняння регресії : y = {} + ({})*x1 + ({})*x2 ".format("%.2f" % B0, "%.2f" % B1, "%.2f" % B2))
print("B0 - B1 - B2 = {} = Y1 = {}".format("%.2f" % (B0 - B1 - B2), average_Y1))
print("B0 + B1 - B2 = {} = Y2 = {}".format("%.2f" % (B0 + B1 - B2), average_Y2))
print("B0 - B1 + B2 = {} = Y3 = {}".format("%.2f" % (B0 - B1 + B2), average_Y3))
print("Результати збігається з середніми значеннями Yj !")
print("3) Натуралізація коефіцієнтів")
print("Δx1 = {}  Δx2 = {}  X10 = {}  X20 = {}".format(delta_x1, delta_x2, x10, x20))
print("a0 = {}  a1 = {}  a2 = {}".format("%.2f" % a2_0, "%.2f" % a2_1, "%.2f" % a2_2))
print("Натуралізоване рівняння регресії: y = {} + ({})*x1 + ({})*x2 ".format("%.2f" % a2_0, "%.2f" % a2_1, "%.2f" % a2_2))
print("Перевірка по рядках:")
print("a2_0 + a2_1*X1_min + a2_2*X2_min = {} = Y1 = {}".format("%.2f" % (a2_0 + a2_1*X1_min + a2_2*X2_min), average_Y1))
print("a2_0 + a2_1*X1_max + a2_2*X2_min = {} = Y2 = {}".format("%.2f" % (a2_0 + a2_1*X1_max + a2_2*X2_min), average_Y2))
print("a2_0 + a2_1*X1_min + a2_2*X2_max = {} = Y3 = {}".format("%.2f" % (a2_0 + a2_1*X1_min + a2_2*X2_max), average_Y3))
print("Отже, коефіцієнти натуралізованого рівняння регресії вірні")
print("\nВиконав: студент групи ІО-92 Іванов Родіон    Варіант 209")







