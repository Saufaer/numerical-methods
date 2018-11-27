#Autor: Lalykin Oleg

#installation commands:
#pip install py_expression_eval
#pip install tabulate
#pip install matplotlib

import math
from tabulate import tabulate
from py_expression_eval import Parser
import matplotlib.pyplot as plt


parser = Parser()

expr = parser.parse("x^2")
A = -10
B = 10
EPS = 0.00001
SIGMA = 0.001
N = 50

def Calc_F(x):
    return expr.evaluate({'x': x})
x1_D=0
x2_D=0
x1_F=0
x2_F=0
dictionary_D = {0:[0,0,0,0,0]}
dictionary_F = {0:[0,0,0,0,0]}

def Plot_F(res_x,res_f):

 data_x = []
 data_y= []
 x = A+A/2
 while x < B+B/2:
    x = x + 0.05
    data_x.append(x)
    data_y.append(Calc_F(x))
 plt.plot(data_x, data_y,"r",label='funtion')
 plt.plot(res_x, res_f,"D",label='result')
 plt.legend()
 plt.ylabel('F(x)')
 plt.xlabel('x')
 plt.show()

     
def dichotomy(func, a, b, eps,sigma):
    steps=0
    left = a
    right = b     
    while right - left > sigma:
        steps = steps + 1
        x1 = (left + right) / 2 - eps
        x2 = (left + right) / 2 + eps
 
        value1 = func(x1)
        value2 = func(x2)
        dictionary_D[steps] = [ steps,x1,value1,x2,value2]
	
	
        if value1 < value2:
            right = x2
        else:
            left = x1
    global  x1_D
    global  x2_D
    x1_D = x1 
    x2_D = x2
    return (left + right) / 2
 
 
def val_fib(n):
    f = 0
    f1 = 1
    f2 = 1 
    m = 0
    while m < (n-1):  
      f = f1 + f2
      f1 = f2
      f2 = f
      m = m + 1
     
    return f1
 
 
def fibonacci(func, a, b, eps,n):
    f_n_plus_2 = (b - a) / eps  
    
    steps=0
    left = a
    right = b
 
    x1 = left + val_fib(n)/val_fib(n + 2) * (b - a)
    x2 = left + val_fib(n + 1)/val_fib(n + 2) * (b - a)
 
    value1 = func(x1)
    value2 = func(x2)
    dictionary_F[steps] = [ steps,x1,value1,x2,value2]
    k = 0
 
    while n > k:
        steps = steps + 1
        k += 1
 
        if value1 < value2:
            right = x2
        else:
            left = x1
 
        x1 = left + val_fib(n - k + 1) / val_fib(n - k + 3) * (right - left)
        x2 = left + val_fib(n - k + 2) / val_fib(n - k + 3) * (right - left)
 
        value1 = func(x1)
        value2 = func(x2)
        dictionary_F[steps] = [ steps,x1,value1,x2,value2]
    global  x1_F
    global  x2_F
    x1_F = x1 
    x2_F = x2
    return (left + right) / 2




def run_dit():
     print('-----------------DICHOTOMY------------------')
     min_dichotomy = dichotomy(Calc_F, A, B, EPS,SIGMA)
     print('Start borders:','\nx1* = ',A,'\nx2* = ',B)
     print('Final borders:','\nx1* = ',x1_D,'\nx2* = ',x2_D)
     print('RESULT:\n','x* = ',min_dichotomy,'\n','minF(x)= ', Calc_F(min_dichotomy),'\n')    
     v = list(dictionary_D.values())
     dictionary_D.clear() 
     clust_data =v
     print (tabulate(v, headers=['step','x1','f(x1)','x2','f(x2)'], tablefmt='orgtbl'))
     Plot_F(min_dichotomy,Calc_F(min_dichotomy))
     
def run_fib():
     print('-----------------FIBONACCI------------------')    
     min_fibonacci = fibonacci(Calc_F, A, B, EPS,N)
     print('Start borders:','\nx1* = ',A,'\nx2* = ',B)
     print('Final borders:','\nx1* = ',x1_F,'\nx2* = ',x2_F)
     print('RESULT:\n','x* = ',min_fibonacci,'\n','minF(x)= ', Calc_F(min_fibonacci),'\n') 
     v = list(dictionary_F.values())
     dictionary_F.clear() 
     clust_data = v
     print (tabulate(v, headers=['step','x1','f(x1)','x2','f(x2)'], tablefmt='orgtbl'))
     Plot_F(min_fibonacci,Calc_F(min_fibonacci))

   

def input_dit():
 global SIGMA 
 SIGMA = float(input("Enter Sigma = "))

def input_fib():
 global N
 N = int(input("Enter number of steps: n = "))

def input_par():
 global expr 
 global A 
 global B 
 global EPS 
 expr = parser.parse(input("Enter function: F(x) = "))
 A = int(input("Enter left border: a = "))
 B = int(input("Enter right border: b = "))
 EPS = float(input("Enter Eps = "))
 
def switch_met(y):
    if y == '1':
        input_dit()
        if SIGMA <= EPS:
            print("ERROR: must be Sigma >> EPS > 0")
            return 7
        run_dit()        
        return 1
    if y == '2':
        input_fib()
        run_fib()
        return 1
    if y == '3':
        return 7
 
def switch_main(x):
    if x == '1':
        input_par()
        return 1
    if x == '2':
        k=0
        while k!=7:
             print('Select method number: \n 1-DICHOTOMY \n 2-FIBONACCI \n 3-Back')
             k = switch_met(input())             
        return 1  
    if x == '3':
        return 5
    
r=0
while r!=5:
    print('MAIN:\n 1-Task input  \n 2-Select method \n 3-Exit')
    r = switch_main(input())
