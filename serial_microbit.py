import serial
from matplotlib import pyplot
import numpy as np
from matplotlib import pyplot
import cvxopt
from cvxopt import matrix
import control

ser = serial.Serial(
      port = "/dev/tty.usbmodem141302", #windows : "COM7"(一番右), mac : "/dev/tty.usbmodem14302"
      baudrate = 115200,
      #parity = serial.PARITY_NONE,
      #bytesize = serial.EIGHTBITS,
      #stopbits = serial.STOPBITS_ONE,
      #timeout = None,
      #xonxoff = 0,
      #rtscts = 0,
      )

# 行列の内積の多重計算の自作関数
def mat_power(A, num):
    result = A
    for m in range(num-1):
        result = np.dot(result, A)
    return result

# Hildrethの方法（双対問題の計算のみ）
def dual_calc(D, c, len):
    x = np.zeros((len, 1))
    count = 0
    x_before = np.zeros(len)
    while True :
        # ギャップの算出・前段階の結果の格納
        gap = 0
        for i in range(len) :
            gap = abs(x[i][0] - x_before[i]) if abs(x[i][0] - x_before[i]) > gap else gap
            x_before[i] = x[i][0]

        # 最適性判定
        if count == 300 or gap < 1e-2 :
            if count != 0 :
                break


        # 解の更新
        wk = np.zeros(len)
        for i in range(len) :
            sum = 0
            for j in range(len) :
                if i == j :
                    continue
                else :
                    sum += D[i][j] * x[j][0]

            wk[i] = -(sum + c[i][0]) / D[i][i]
            x[i][0] = wk[i] if wk[i] > 0 else 0

        count += 1

    return x

def dual(H, q, G, Cu):
    # 双対問題の行列計算
    D = G@np.linalg.inv(H)@(G.T)
    c = G@np.linalg.inv(H)@q
    for i in range(c.shape[0]//2) :
        c[2 * i][0] += Cu[0][0]
        c[2 * i + 1][0] += Cu[1][0]

    # 最適解の導出
    x = dual_calc(D, c, G.shape[0])
    u = - np.linalg.inv(H)@(q + (G.T)@x)

    # Xの算出(二つに分ける)
    Xa = np.empty(Np+1)
    Xb = np.empty(Np+1)
    X1 = np.dot(A_, x0) 
    X2 = np.dot(B_, u)
    for i in range(Np+1):
        Xa[i] = X1[2*i] + X2[2*i]
        Xb[i] = X1[2*i+1] + X2[2*i+1]

    return u, Xa, Xb

def optimization(H, q, G, Cu):
    # hの算出
    h = np.empty([2*Np, 1])
    for i in range(0, Np):  # hの設定
        h[2*i, 0] = Cu[0][0]
        h[2*i+1, 0] = Cu[1][0]

    # qp問題を解く関数に用いる係数のmatrix設定
    P = matrix(H) # ソルバーに利用するためmatrixにしておく
    q = matrix(q) # 同上
    G = matrix(G) # 同上
    h = matrix(h) # 同上

    sol = cvxopt.solvers.qp(P, q, G, h)  # 最適化
    U = sol["x"]

    # Xの算出(二つに分ける)
    Xa = np.empty(Np+1)
    Xb = np.empty(Np+1)
    X1 = np.dot(A_, x0) 
    X2 = np.dot(B_, sol["x"])
    for i in range(Np+1):
        Xa[i] = X1[2*i] + X2[2*i]
        Xb[i] = X1[2*i+1] + X2[2*i+1]
    
    return U, Xa, Xb

################################################################

#初期設定

#予測区間
Np = 30

# 状態初期値
x0 = np.array([[1.0], [-1.5]]) #1.0 1.3

# パラメータA, Bの設定
A = np.array([[1, 0.2], [0, 1]])  # control.dareで使用するためmatrixに
B = np.array([[0.02], [0.2]])  # control.dareで使用するためにmatrixに

#制約条件の設定
Fu = np.array([[1], [-1]])
Cu = np.array([[1], [1]])

# その他パラメータの設定
R = 0.1
Q = np.diag([1,1])

###################################################################

#最適化問題Pの設定

# Q_の算出
Q_data = np.empty(2*(Np+1)) * 0
for i in range(0, Np):  # Q_の要素の代入
    Q_data[2*i] = Q[0, 0]
    Q_data[2*i+1] = Q[1, 1]

Q_ = np.diag(Q_data)  # Q_の作成

Qf, L, K = control.dare(A, B, Q, R)  # 離散時間リカッチ方程式の解(解,固有値,ゲイン)

for i in range(0,2):
    for j in range(0,2):
        Q_[2*Np+i, 2*Np+j] = Qf[i, j]  # Q_にQfを要素として代入


# R_の算出
R_data = np.empty(Np)
for i in range(0,Np):
    R_data[i] = R
R_ = matrix(np.diag(R_data))


# A_の算出
A_ = np.empty((2*(Np+1), 2)) * 0
for i in range(Np, -1, -1):  # A_の作成
    if i > 0:
        tmp = mat_power(A, i)
        A_[2*i, 0] = tmp[0, 0]
        A_[2*i, 1] = tmp[0, 1]
        A_[2*i+1, 0] = tmp[1, 0]
        A_[2*i+1, 1] = tmp[1, 1]
    else:
        A_[2*i, 0] = 1
        A_[2*i+1, 1] = 1


# B_の作成
B_ = np.empty((2*(Np+1),Np)) * 0
for i in range(Np, -1, -1):  # 2iが行
    for j in range(0, Np):  # jが列
        if i-1-j > 0:
            tmp = np.dot(mat_power(A, i-1-j), B)
            B_[2*i, j] = tmp[0, 0]
            B_[2*i+1, j] = tmp[1, 0]
        elif i-1-j == 0:
            B_[2*i, j] = B[0, 0]
            B_[2*i+1, j] = B[1, 0]
        else:
            break

# H, F, qの算出
H = np.array(2*(R_ + np.dot(np.dot(B_.T, Q_), B_))) 
F = np.array(2*(np.dot(np.dot(A_.T, Q_), B_)))  
qT = np.dot(x0.T, F)
q = qT.T

# Gの算出
G = np.empty((2*Np, Np)) * 0
# Gの上半分
for i in range(0, Np):  # 2iが行
    for j in range(0, Np):  # 列
        if i < Np and i == j:
            G[2*i, j] = Fu[0]
            G[2*i+1, j] = Fu[1]

u, firstX, secondX = optimization(H, q, G, Cu)

#######################################################

U = np.zeros(Np)
count = 0

def plot(U, u, firstX, secondX):
    # 相対誤算
    U = U.reshape(-1,1)
    gap = 0
    gap1 = 0
    gap2 = 0
    #for i in range(Np):
        #gap1 += (U[i]-u[i]) ** 2
        #gap2 += u[i] ** 2
    gap = np.linalg.norm(U-u, ord=2) / np.linalg.norm(u, ord=2) * 100
    print('相対誤差 : ', gap)
    #print('||U-u|| : ', np.linalg.norm(U-u, ord=2))
    #print('||u|| : ', np.linalg.norm(u, ord=2))
    #print(U-u)
    #print('相対誤差 : ', np.sqrt(gap1) / np.sqrt(gap2) * 100)

    # Xの算出(二つに分ける)
    Xa = np.zeros(Np+1)
    Xb = np.zeros(Np+1)
    X1 = np.dot(A_, x0) 
    X2 = np.dot(B_, U)

    for i in range(Np+1):
        Xa[i] = X1[2*i] + X2[2*i]
        Xb[i] = X1[2*i+1] + X2[2*i+1]

    # グラフのフォント選定
    pyplot.rcParams['pdf.fonttype'] = 42
    pyplot.rcParams['ps.fonttype'] = 42
    pyplot.rcParams["font.size"] = 16

    # 入力の最適解のグラフ設定
    pyplot.rcParams["font.size"] = 16
    fig1 = pyplot.figure()
    ax1 = fig1.add_subplot(111)
    t = np.zeros(Np)
    for i in range(Np):
        t[i] = i
    ax1.step(t, u, where='post', label = 'Exact', color = 'blue', linestyle="--")
    ax1.step(t, U, where='post', label='d=3', color='red') #red
    ax1.set_xlabel('Prediction step k')
    ax1.set_ylabel('Input uk')
    ax1.legend()
    fig1.tight_layout()

    # 状態の最適解のグラフ設定
    fig2 = pyplot.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(firstX, label = 'Exact', color = 'blue', linestyle = "--")
    ax2.plot(secondX, color = 'blue', linestyle = "--")
    ax2.plot(Xa, label='d=3', color='red') #red
    ax2.plot(Xb, color='red') #red
    ax2.set_xlabel('Prediction step k')
    ax2.set_ylabel('State xk')
    ax2.legend()
    fig2.tight_layout()

    pyplot.show()  # グラフ表示


ser.reset_input_buffer()

while True:
    recv_data = ser.readline()
    if recv_data != b'':
        if recv_data == b'\r\n':
            plot(U, u, firstX, secondX)
            break
        U[count] = float(recv_data.rstrip().decode('utf-8'))
        print(U[count])
        count += 1

ser.close()