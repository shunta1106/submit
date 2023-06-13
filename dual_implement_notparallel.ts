// 使用機器　：　v2yellow
/* 元の最適化問題　：　min 1/2 u^T@H@u + d^T@u
                    s.t. A@u <= b

   元の最適化問題に対する双対問題 : min 1/2 x^T@D@x + c^T@x
                               s.t. x >= 0

    ただし、D = A@H^(-1)@A^T,  c = b + A@H^(-1)@d,  u = -H^(-1)@(d+A^T@x)

    mpc反復で変化する行列はd, b (Xpを相互交換するため)
    d = (Xp^T@F)^T,  b = W + E@Xp(状態制約がない時 : b = W)

    状態 : X = A_@Xp + B_@u

    in  : F, W(cu), Xp, H^(-1), A, A_, B_
          (状態制約を消去した時)
    out : u(入力), X_bs(t=bs時の状態)
*/

// 2023/01/27 作成
//このプログラムは動かない！(エラー020)
//2023/01/27時点では, MPCは分散実装しないとダメ

/*****************************************************/
export{}

// 予測ホライゾンの設定
const Np: number = 30;

// パラメータA, Bの設定
const A: number[][] = [[1, 0.2], [0, 1]];
const B: number[][] = [[0.02], [0.2]];

//制約条件の設定
const Fu: number[][] = [[1], [-1]];
const Cu: number[][] = [[1], [1]];

// 初期状態の設定
const x0: number[][] = [[1.0], [1.3]];

// その他パラメータの設定
const R: number = 0.1;
const Q: number[][] = diag([1, 1]);
const Qf: number[][] = [[6.968, 1.658], [1.658, 2.645]];  // 離散時間リカッチ方程式の解

/***********************************************************************/

// 絶対値を算出する関数
function abs(num: number) {
    if (num >= 0) {
        return num;
    } else {
        return -num;
    }
}

// 行列の連結（行・列方向）
function concat_Array(array1: number[][], array2: number[][], axis: number) {
    if (axis != 1) {
        axis = 0;
    }
    let array3: number[][] = [];
    if (axis == 0) {  //　縦方向の結合
        array3 = array1.slice();
        for (let i = 0; i < array2.length; i++) {
            array3.push(array2[i]);
        }
    }
    else {  //　横方向の結合
        for (let i = 0; i < array1.length; i++) {
            array3[i] = array1[i].concat(array2[i]);
        }
    }
    return array3;
}

//行列の行数(row)、列数(col)を調べる。[row, col]を返す。
function is_matrix(array: number[][]): number[] {
    let row = array.length;
    let col = array[0].length;
    return [row, col];
}

//零行列
function zeros(row: number, col: number): number[][] {
    let zeros: number[][] = [];
    for (let i = 0; i < row; i++) {
        let tmp = [];
        for (let j = 0; j < col; j++) {
            tmp.push(0);
        }
        zeros.push(tmp);
    }
    return zeros;
}

//単位行列
function eye(row: number, col: number): number[][] {
    let eye: number[][] = [];
    for (let i = 0; i < row; i++) {
        let tmp: number[] = [];
        for (let j = 0; j < col; j++) {
            if (i == j) {
                tmp.push(1);
            } else {
                tmp.push(0);
            }
        }
        eye.push(tmp);
    }
    return eye;
}

// 配列のすべての要素を同じ数字に初期化
function initialize(length: number, x: number): number[] {
    let initial: number[] = [];
    for (let i = 0; i < length; i++) {
        initial[i] = x;
    }
    return initial;
}

//対角行列
function diag(c: number[]): number[][] {
    let diag: number[][] = [];
    for (let i = 0; i < c.length; i++) {
        let tmp: number[] = [];
        for (let j = 0; j < c.length; j++) {
            if (i !== j) {
                tmp.push(0);
            } else {
                tmp.push(c[i]);
            }
        }
        diag.push(tmp);
    }
    return diag;
}

//行列の和 A+B
function add_asmatrix(A: number[][], B: number[][]): number[][] {
    let Amn: number[] = is_matrix(A);
    let Bmn: number[] = is_matrix(B);
    let tmp: number[][] = zeros(Amn[0], Amn[1]);
    for (let i = 0; i < Amn[0]; i++) {
        for (let j = 0; j < Amn[1]; j++) {
            tmp[i][j] = A[i][j] + B[i][j];
        }
    }
    return tmp;
}

//行列のスカラー倍 xA
function scalar_mul(A: number[][], x: number) {
    let mn: number[] = is_matrix(A);
    let tmp: number[][] = zeros(mn[0], mn[1]);
    for (let i = 0; i < mn[0]; i++) {
        for (let j = 0; j < mn[1]; j++) {
            tmp[i][j] = A[i][j] * x;
        }
    }
    return tmp;
}

//行列の転置 A^t
function matrix_T(A: number[][]): number[][] {
    let mn: number[] = is_matrix(A);
    let tmp: number[][] = zeros(mn[1], mn[0]);
    for (let i = 0; i < mn[0]; i++) {
        for (let j = 0; j < mn[1]; j++) {
            tmp[j][i] = A[i][j];
        }
    }
    return tmp;
}

//行列の内積 <A,B>
function inner_prod(A: number[][], B: number[][]): any {
    let Amn: number[] = is_matrix(A);
    let Bmn: number[] = is_matrix(B);
    let tmp: number[][] = [];
    for (let i = 0; i < Amn[0]; i++) {
        tmp.push([]);
        for (let j = 0; j < Bmn[1]; j++) {
            tmp[i].push(0);
            for (let k = 0; k < Amn[1]; k++) {
                tmp[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return tmp;
}

// 行列の内積の多重計算の自作関数
function mat_power(A: number[][], num: number): number[][] {
    let result: number[][] = A;
    for (let m = 0; m < num - 1; m++) {
        result = inner_prod(result, A);
    }
    return result;
}

/*********************************************************************************************/

//最適化問題Pの設定

//Q_の算出
let Q_data: number[] = initialize(2 * (Np+1), 0);
for (let i = 0; i < Np; i++) {  // Q_の要素の代入
    Q_data[2 * i] = Q[0][0];
    Q_data[2 * i + 1] = Q[1][1];
}
let Q_: number[][] = diag(Q_data);  // Q_の作成
for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
        Q_[2 * Np + i][2 * Np + j] = Qf[i][j];  // Q_にQfを要素として代入
    }
}

//R_の算出
let R_data: number[] = initialize(Np, R);
const R_: number[][] = diag(R_data);

// A_の算出
let A_: number[][] = zeros(2 * (Np + 1), 2);
for (let i=Np; i>-1; i--) {
    if (i > 0) {
        const tmp: number[][] = mat_power(A, i);
        A_[2 * i][0] = tmp[0][0];
        A_[2 * i][1] = tmp[0][1];
        A_[2 * i + 1][0] = tmp[1][0];
        A_[2 * i + 1][1] = tmp[1][1];
    } else {
        A_[2 * i][0] = 1;
        A_[2 * i + 1][1] = 1;
    }
}
console.log('A_ : ');
console.log(A_);

// B_の作成
let B_: number[][] = zeros(2 * (Np + 1), Np);
for (let i=Np; i>-1; i--) {
    for (let j=0; j<Np; j++) {
        if (i - 1 - j > 0) {
            const tmp: number[][] = inner_prod(mat_power(A, i - 1 - j), B);
            B_[2 * i][j] = tmp[0][0];
            B_[2 * i + 1][j] = tmp[1][0];
        } else if (i - 1 - j == 0) {
            B_[2 * i][j] = B[0][0];
            B_[2 * i + 1][j] = B[1][0];
        } else {
            break;
        }
    }
}
console.log('B_ : ');
console.log(B_);

//H, Fの算出
const H: number[][] = scalar_mul(add_asmatrix(R_, inner_prod(inner_prod(matrix_T(B_), Q_), B_)), 2);
console.log('H : ');
console.log(H);
const F: number[][] = scalar_mul(inner_prod(inner_prod(matrix_T(A_), Q_), B_), 2);
console.log('F : ');
console.log(F);

//Gの算出
let G: number[][] = zeros(2 * Np, Np);
// Gの上半分のみ
for (let i = 0; i < Np; i++) { // 2iが行
    for (let j = 0; j < Np; j++) {  // 列
        if (i < Np && i == j) {
            G[2 * i][j] = Fu[0][0];
            G[2 * i + 1][j] = Fu[1][0];
        }
    }
}

function GaussJordan(A: number[][]) {
    // 拡張行列の右半分を単位行列にする
    const n: number = is_matrix(A)[0];
    A = concat_Array(A, eye(n, n), 1);

    // ガウス・ジョルダン法(Gauss-Jordan method)で逆行列計算
    for (let k = 0; k < n; ++k) {
        const akk: number = A[k][k];
        // 対角要素を1にするために，k行目のすべての要素をa_kkで割る
        for (let j = 0; j < 2 * n; ++j) {
            A[k][j] /= akk;
        }

        // k列目の非対角要素を0にする
        for (let i = 0; i < n; ++i) {
            if (i == k) continue;
            const aik = A[i][k];
            for (let j = 0; j < 2 * n; ++j) {
                A[i][j] -= A[k][j] * aik;
            }
        }
    }

    // 逆行列部分以外を削除
    let exe: number[] = [];
    for (let i = 0; i < n; i++) {
        exe[i] = i;
    }
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < exe.length; j++) {
            A[i].splice(exe[j] - j, 1);
        }
    }

    return A;
}

// Hildrethの方法（双対問題の計算のみ）
function dual_calc(D: number[][], c: number[][], len: number): number[][] {
    let x: number[][] = zeros(len, 1);
    let count: number = 0;
    let x_before: number[] = initialize(len, 0);
    while (true) {
        // ギャップの算出・前段階の結果の格納
        let gap: number = 0;
        for (let i = 0; i < x_before.length; i++) {
            gap = abs(x[i][0] - x_before[i]) > gap ? abs(x[i][0] - x_before[i]) : gap;
            x_before[i] = x[i][0];
        }

        // 最適性判定
        if (count == 500 || gap < 1e-2) { // 1e-3
            if (count != 0) {
                console.log('反復回数 : '+count);
                break;
            }
        }

        // 解の更新
        let wk: number[] = initialize(len, 0);
        for (let i = 0; i < len; i++) {
            let sum: number = 0;
            for (let j = 0; j < len; j++) {
                if (i == j) {
                    continue;
                } else {
                    sum += D[i][j] * x[j][0];
                }
            }
            wk[i] = -(sum + c[i][0]) / D[i][i];
            x[i][0] = (wk[i] > 0) ? wk[i] : 0;
        }
        count++;
    }
    return x;
}

// 双対問題の行列計算
const D: number[][] = inner_prod(inner_prod(G, GaussJordan(H)), matrix_T(G));

// 最適化
function calculation(D: number[][], F: number[][], W: number[][], Xp: number[][], H_inv: number[][], A: number[][]) {
    let c: number[][] = inner_prod(inner_prod(A, H_inv), matrix_T(inner_prod(matrix_T(Xp), F)));
    for (let i = 0; i < is_matrix(c)[0] / 2; i++) {
        c[2 * i][0] += W[0][0];
        c[2 * i + 1][0] += W[1][0];
    }
    const x: number[][] = dual_calc(D, c, is_matrix(A)[0]);
    const u: number[][] = scalar_mul(inner_prod(H_inv, add_asmatrix(matrix_T(inner_prod(matrix_T(Xp), F)), inner_prod(matrix_T(A), x))), -1);
    console.log('u : ');
    for (let i = 0; i < is_matrix(u)[0]; i++) {
        console.log(u[i][0]);
    }
}

calculation(D, F, Cu, x0, GaussJordan(H), G);
