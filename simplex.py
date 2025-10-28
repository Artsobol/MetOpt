import sys

import numpy as np
import json

EPS = 1e-9

def load_lp(path):
    d = json.load(open(path, "r", encoding="utf-8"))
    return (
        d["sense"].lower(),                 # max | min
        np.array(d["c"], float),            # (n,)
        np.array(d["A"], float),            # (m,n)
        np.array(d["b"], float),            # (m,)
        [s.strip() for s in d["constraints"]]  # [<=, =, >=, ...] длины m
    )

def pivot(T, i, j):
    """
    Поворот по элементу (i, j):
    1) Делим всю ведущую строку i на ведущий элемент T[i, j], делая его равным 1
    2) Вычитаем из остальных строк так, чтобы во всём столбце j, кроме строки i, стали нули
    Новый базисный элемент становится единичным, а столбец j превращается в базисный столбец.
    """
    T[i, :] /= T[i, j]
    for r in range(T.shape[0]):
        if r != i:
            T[r, :] -= T[r, j] * T[i, :]

def choose_entering(T):
    """
    Выбираем самый отрицательный коэффициент в строке цели
    Если отрицательных нет - задача решена
    """
    last = T[-1, :-1]
    j_min, v_min = None, 0.0
    for j, v in enumerate(last):
        if v < -EPS and (j_min is None or v < v_min):
            j_min, v_min = j, v
    return j_min

def choose_leaving(T, j):
    """
    Выбираем базисную строку, где a_ij > 0 и при этом минимально
    """
    b = T[:-1, -1]
    col = T[:-1, j]
    ratios = [b[i] / col[i] if col[i] > EPS else np.inf for i in range(len(b))]
    i = int(np.argmin(ratios))
    return None if np.isinf(ratios[i]) else i

def set_objective_coef(T, basis, cvec):
    """
    Формируем последнюю строку
    """
    m, n_plus_rhs = T.shape[0]-1, T.shape[1]
    n = n_plus_rhs - 1
    obj = np.zeros(n_plus_rhs, float)
    obj[:n] = cvec
    obj[-1] = 0.0
    for i in range(m):
        bcol = basis[i]
        cb = cvec[bcol] if 0 <= bcol < n else 0.0
        if abs(cb) > EPS:
            obj -= cb * T[i, :]
    T[-1, :] = -obj

def simplex(T, basis, max_iters=10000):
    """
    Выбираем сначала минимальный столбец, потом строку, меняем базис
    """
    for _ in range(max_iters):
        j = choose_entering(T)
        if j is None:
            return "ok"
        i = choose_leaving(T, j)
        if i is None:
            return "unbounded"
        pivot(T, i, j)
        basis[i] = j
    return "ok"

def normalize(A, b, cons):
    """
    Приводим к каноническому виду
    """
    A2, b2, c2 = A.copy(), b.copy(), cons[:]
    m = len(b2)

    for i in range(m):
        if b2[i] < -EPS:
            A2[i, :] *= -1
            b2[i] *= -1
            if c2[i] == "<=": c2[i] = ">="
            elif c2[i] == ">=": c2[i] = "<="

    for i in range(m):
        if c2[i] == ">=":
            A2[i, :] *= -1
            b2[i] *= -1
            c2[i] = "<="

    return A2, b2, c2


def build_table(A, b, cons):
    """
    Сборка таблицы
    """
    m, n = A.shape
    rows = [A[i, :].tolist() for i in range(m)]
    var_types = ["x"] * n
    basis = [-1] * m

    def add_col(col_vals, kind):
        for i in range(m):
            rows[i].append(col_vals[i])
        var_types.append(kind)
        return len(var_types) - 1

    for i in range(m):
        if cons[i] != "=":
            col = [0.0]*m; col[i] = 1.0
            s = add_col(col, "s")
            basis[i] = s

    n_tot = len(var_types)
    T = np.zeros((m+1, n_tot+1), float)
    for i in range(m):
        T[i, :n_tot] = rows[i]
        T[i, -1] = b[i]

    info = {
        "basis": basis,
        "var_types": var_types,
        "n_original": n
    }
    return T, info


def try_fill_missing_basis(T, basis, n_orig):
    """
    Поиск базиса
    """
    m, n_plus_rhs = T.shape[0]-1, T.shape[1]
    n_all = n_plus_rhs - 1

    variables = []
    x_idxs = list(range(n_orig))
    for j in range(n_orig, n_all):
        variables.append(j)

    for i in range(m):
        if basis[i] != -1:
            continue

        jcand = None

        for j in variables + x_idxs:
            if abs(T[i, j]) > EPS:
                jcand = j
                break

        if jcand is None:
            return False 

        pivot(T, i, jcand)
        basis[i] = jcand

    return True


def set_objective(T, basis, var_types, n_orig, c_max):
    """
    Заполняем коэффициенты
    """
    n_plus_rhs = T.shape[1]
    n_all = n_plus_rhs - 1
    c_ext = np.zeros(n_all)
    for j in range(n_all):
        if j < n_orig and var_types[j] == "x":
            c_ext[j] = c_max[j]
        else:
            c_ext[j] = 0.0
    set_objective_coef(T, basis, c_ext)


def solve(path):
    """
    Решение задачи
    """
    sense, c, A, b, cons = load_lp(path)
    is_min = (sense == "min")
    c_max = -c if is_min else c

    A1, b1, cons1 = normalize(A, b, cons)

    T, info = build_table(A1, b1, cons1)

    ok = try_fill_missing_basis(T, info["basis"], info["n_original"])
    if not ok:
        return {"status": "infeasible", "message": "Базис не найден"}

    set_objective(T, info["basis"], info["var_types"], info["n_original"], c_max)

    status = simplex(T, info["basis"])
    if status == "unbounded":
        return {"status": "unbounded", "message": "Целевая функция неограничена"}

    n = info["n_original"]; m = T.shape[0]-1
    x = np.zeros(n)
    for i in range(m):
        bcol = info["basis"][i]
        if 0 <= bcol < n:
            x[bcol] = T[i, -1]

    z_max = T[-1, -1]
    z = -z_max if is_min else z_max

    x = [int(round(v)) if abs(v - round(v)) < 1e-9 else float(round(v, 10)) for v in x]
    z = float(z); z = int(round(z)) if abs(z - round(z)) < 1e-9 else round(z, 10)

    return {"status": "optimal", "x_opt": x, "objective": z}

def result(res):
    st = res.get("status", "")
    if st == "optimal":
        print("Найдено решение:")
        for i, v in enumerate(res["x_opt"], 1):
            print(f"x{i} = {v}")
        print(f"Z = {res['objective']}")
    elif st == "infeasible":
        print("Решение не найдено: нет допустимых решений.")
        msg = res.get("message")
        if msg: print("Причина:", msg)
    elif st == "unbounded":
        print("Решение не найдено: целевая функция неограничена.")
        msg = res.get("message")
        if msg: print("Детали:", msg)
    else:
        print("Неизвестный статус:", res)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python simplex.py <path_to_json>")
        sys.exit(1)
    res = solve(sys.argv[1])
    result(res)
