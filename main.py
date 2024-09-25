import numpy as np
from itertools import combinations

# Генерация случайной размерности матрицы
rows = np.random.randint(5, 11)  # случайное количество строк от 5 до 10
cols = np.random.randint(5, 11)  # случайное количество столбцов от 5 до 10

# Генерация матрицы, состоящей из нулей и единиц
matrix = np.random.randint(0, 2, size=(rows, cols))

#Матрица из примера

example_G = np.array([[1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
                      [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
                      [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])


def REF(matrix):
    mat = matrix.astype(int)
    rows, cols = mat.shape
    lead_col = 0  # Столбец ведущего элемента

    for row in range(rows):
        if lead_col >= cols:
            return mat

        i = row
        while mat[i, lead_col] == 0:
            i += 1
            if i == rows:
                i = row
                lead_col += 1
                if lead_col == cols:
                    return mat

        # Меняем местами текущую строку и строку с ненулевым элементом
        mat[[i, row]] = mat[[row, i]]

        # Обнуляем элементы под ведущим элементом (сложение по модулю 2)
        for i in range(row + 1, rows):
            if mat[i, lead_col] == 1:
                mat[i] = (mat[i] + mat[row]) % 2

        lead_col += 1

    return mat

print(f"Размер матрицы: {rows}x{cols} \n")

print(f"Изначальная матрица: \n {example_G} \n")
ref = REF(example_G)
print(f"Матрица ступенчатого вида: \n {ref} \n")


def RREF(matrix):
    mat = matrix.astype(int)
    rows, cols = mat.shape
    lead_col = 0

    for row in range(rows):
        if lead_col >= cols:
            return mat

        i = row
        while mat[i, lead_col] == 0:
            i += 1
            if i == rows:
                i = row
                lead_col += 1
                if lead_col == cols:
                    return mat

        #Меняем строки, если ведущий элемент не на своем месте
        mat[[i, row]] = mat[[row, i]]


        #Обнуляем элементы в столбце, не трогая ведущий элемент
        for i in range(rows):
            if i != row and mat[i, lead_col] == 1:
                mat[i] = (mat[i] + mat[row]) % 2

        lead_col += 1

    return mat


print(f"Размер матрицы: {rows}x{cols} \n")

print(f"Изначальная матрица: \n {example_G} \n")
rref = RREF(example_G)
print(f"Приведенная матрица ступенчатого вида: \n {rref} \n")

k = example_G.shape[0]
n = example_G.shape[1]
def fix_lead_cols(matrix):
    rows, cols = matrix.shape
    lead_cols = []

    for row in range(rows):
        for col in range(cols):
            if matrix[row, col] == 1:
                lead_cols.append(col)
                break  # Переходим к следующей строке, как только нашли ведущую единицу

    return lead_cols

lead_cols = fix_lead_cols(rref)



print("Ведущие столбцы: \n")
print(lead_cols, "\n")

def delete_lead_cols(matrix, lead_cols):
    return np.delete(matrix, lead_cols, axis=1)

X = delete_lead_cols(rref, lead_cols)

print("Сокращённая матрица: \n", X, "\n")


I = np.eye(len(X[0]), dtype=int)
print("Единичная матрица: \n", I)

def H_matrix_create(X, lead_cols, I):
    cols = np.shape(X)[1]
    rows = np.shape(X)[0] + np.shape(I)[0]
    strX = 0
    strI = 0
    H = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        if i in lead_cols:
            H[i, :] = X[strX, :]
            strX += 1
        else:
            H[i, :] = I[strI, :]
            strI += 1

    return H

H = H_matrix_create(X, lead_cols, I)
print("Матрица H: \n", H, "\n")


def codewords_by_sum(example_G, k):
    codewords = set()

    for r in range(1, k + 1):
        for comb in combinations(range(k), r):

            codeword = np.bitwise_xor.reduce(example_G[list(comb)], axis=0)
            codewords.add(tuple(codeword))

    codewords.add(tuple(np.zeros(example_G.shape[1], dtype=int)))

    return np.array(list(codewords))

sum_codewords = codewords_by_sum(example_G, k)
print("Все кодовые слова (способ 1): \n", sum_codewords,"\n")
def codewords_by_multiplication(G, k):
    codewords = []

    # Генерируем все двоичные слова длины k
    for i in range(2**k):
        binary_word = np.array(list(np.binary_repr(i, k)), dtype=int)
        codeword = np.dot(binary_word, G) % 2
        codewords.append(codeword)

    return np.array(codewords)

mult_codewords = codewords_by_multiplication(example_G, k)
print("Все кодовые слова (способ 2): \n", mult_codewords,"\n")


print({tuple(word.tolist()) for word in sum_codewords} == {tuple(word.tolist()) for word in mult_codewords})

def distance(sum_codewords):
    min_weight = float('inf')
    for word in sum_codewords:
        weight = np.sum(word)
        if 0 < weight < min_weight:
            min_weight = weight
    return min_weight

print("\n Кодовое расстояние:",distance(sum_codewords), "\n")

error = np.zeros_like(sum_codewords[0])
error[4] = 1
codeword_with_error = error + sum_codewords[22]
xor_error_codeword = codeword_with_error % 2

def check(codeword, H):
    return np.dot(codeword, H) % 2

print("Убеждаемся в обнаружении ошибки: ", check(xor_error_codeword, H))

error_2 = np.zeros_like(sum_codewords[0])
error_2[6] = 1
error_2[9] = 1
codeword_with_error_2 = error_2 + sum_codewords[22]
xor_error_codeword_2 = codeword_with_error_2 % 2

print("Убеждаемся в отсутствии ошибки: ", check(xor_error_codeword_2, H))







