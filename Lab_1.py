import random
import timeit
start_time = timeit.default_timer()

A0, A1, A2, A3 = [random.randint(0, 20) for j in range(4)]

#заповнення матриці випадковим чином
def random_x():
    return [random.randint(0, 20) for i in range(8)]
x1, x2, x3 = [random_x() for k in range(3)]

#обчислюємо значення функції відгукув для кожної точки плану за формулою лінійної регресії
def calculate_Y(x1, x2, x3):
    return A0 + A1 * x1 + A2 * x2 + A3 * x3

#обчислюємо значення X0 для кожного фактора
def calculate_x0i(x_results):
    return (max(x_results) + min(x_results)) / 2

#обчислюємо інтервал зміни фактора
def calculate_dxi(x0i, x_results):
    return x0i - min(x_results)

#знаходимо нормаване значенне Xn для кожного фактора
def calculate_xni(x0i, dxi, x_results):
    return [((i - x0i) / dxi) for i in x_results]

#пошук точки плану, що задоволняє криторію вибору оптимальності
# -------------------------------------------------------------- #
def average_Y(Y):
    s = 0
    for i in Y:
        s += i
    return s/len(Y)

def optimal(a_Y, Y):
    opt = []
    for i in range(8):
        opt.append(Y[i] - a_Y)
    return opt

def check(optimal):
    return min((a,i) for i, a in enumerate(optimal) if a>0)[1]
# -------------------------------------------------------------- #


Y = [calculate_Y(x1[i], x2[i], x3[i]) for i in range(8)]

X01 = calculate_x0i(x1)
X02 = calculate_x0i(x2)
X03 = calculate_x0i(x3)

Dx1 = calculate_dxi(X01, x1)
Dx2 = calculate_dxi(X02, x2)
Dx3 = calculate_dxi(X03, x3)

Xn1 = calculate_xni(X01, Dx1, x1)
Xn2 = calculate_xni(X02, Dx2, x2)
Xn3 = calculate_xni(X03, Dx3, x3)

# обчислюємо функцію відгуку від нульових рівнів факторів, еталонне Yет
Y2 = calculate_Y(X01, X02, X03)

a_Y = average_Y(Y)
opt = optimal(a_Y, Y)
index = check(opt)
OPT_POINT = [x1[index], x2[index], x3[index]]

print("A0 = {0}  A1 = {1}  A3 = {2}  A4 = {3}".format(A0, A1, A2, A3))
print("-"*61)
print("N | X1   X2   X3  |   Y3    |         | Xn1    Xn2    Xn3   |")
print("-"*61)
for i in range(8):
    print(f"{i+1:^1} |{x1[i]:^4} {x2[i]:^4} {x3[i]:^4} |"
          f"{Y[i]:^7}  |"
          f"{'%.2f' %opt[i]:^8} | {'%.2f' % Xn1[i]:^5}  {'%.2f' % Xn2[i]:^5}  {'%.2f' % Xn3[i]:^5} |")
print("-"*61)
print(f"X0| {X01:^4} {X02:^4} {X03:^4}| {Y2:^7} |")
print(f"Dx| {Dx1:^4} {Dx2:^4} {Dx3:^4}|")
print(f"\nЕталонне Yет: = {A0} + {A1}*x01 + {A2}*x02 + {A3}*x03" )
print(f"\nФункція: Y = {A0} + {A1}*x1 + {A2}*x2 + {A3}*x3")
print("Оптимальна точка плану(Критерій оптимальності - Yсереднє <-):  Y({0}, {1}, {2}) = {3}".format(*OPT_POINT, "%.1f" % Y[index]))
print("Виконав: студент групи ІО-92 Іванов Родіон    Варіант 209")
print("Час виконання роботи програми дорівнює {} секунд.".format(round(timeit.default_timer() - start_time, 5)))