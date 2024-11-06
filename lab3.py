import itertools
import numpy as np

def create_H_matrix(r):
    n = 2**r - 1
    k = n - r
    H = np.zeros((n, r), dtype=int)

    #последние r строк заполняем единичной матрицей
    H[k : n] = np.eye(r, dtype=int)

    combinations = [list(el) for el in itertools.product([0, 1], repeat=r) if sum(el) > 1]

    H[:k, :] = combinations[:n - r]

    return H

def create_G_matrix(H, r):
    n = 2 ** r - 1
    k = n - r
    G = np.zeros((k, n), dtype=int)

    #первые k столбцов - единичная матрица
    G[:, 0:k] = np.eye(k, dtype=int)

    #последние r столбцов - первые k строк матрицы H
    G[:, k:] = H[:k, :]
    return G

def create_syndrome_table(H):
    syndrome_table = {}
    for i in range(len(H)):
        syndrome_table[np.array_str(H[i, :])] = i
    return syndrome_table

def encode_message(message, G):
    return np.dot(message, G) % 2

def introduce_errors(codeword, error_positions):
    #вносим ошибки в кодовое слово на заданных позициях
    codeword_with_errors = codeword.copy()
    for pos in error_positions:
        codeword_with_errors[pos] ^= 1
    return codeword_with_errors

def compute_syndrome(codeword, H):
    return np.dot(codeword, H) % 2

def correct_errors(codeword, H, syndrome_table):
    syndrome = compute_syndrome(codeword, H)
    syndrome_str = np.array_str(syndrome)

    if syndrome_str in syndrome_table:
        error_position = syndrome_table[syndrome_str]
        print(f"Ошибка на позиции: {error_position}")
        codeword[error_position] ^= 1  #исправляем ошибку
    else:
        print("Ошибок не найдено.")

    return codeword

def check_correction(original, corrected):
    if np.array_equal(original, corrected):
        return "Ошибка исправлена"
    else:
        return "Ошибка не исправлена"

print("\nДля кода Хеммминга")

for r in [2, 3, 4]:
    print(f"\nИсследование для r = {r}")

    H = create_H_matrix(r)
    G = create_G_matrix(H, r)
    syndrome_table = create_syndrome_table(H)

    print("H: \n", H)
    print("G: \n", G)
    print("Таблица синдромов: \n", syndrome_table)

    k = 2**r - 1 - r
    message = np.random.randint(0, 2, k)
    print(f"Сообщение: {message}")

    #кодирование сообщения
    encoded_message = encode_message(message, G)
    print(f"Закодированное сообщение: {encoded_message}")

    codeword_with_1_error = introduce_errors(encoded_message, [0])
    print(f"Сообщение с 1 ошибкой: {codeword_with_1_error}")
    corrected_message = correct_errors(codeword_with_1_error, H, syndrome_table)
    print(f"Исправленное сообщение: {corrected_message}")
    print(check_correction(encoded_message, corrected_message))

    if 3 <= r <= 4:
        codeword_with_2_errors = introduce_errors(encoded_message, [0, 1])
        print(f"Сообщение с 2 ошибками: {codeword_with_2_errors}")
        corrected_message = correct_errors(codeword_with_2_errors, H, syndrome_table)
        print(f"Исправленное сообщение: {corrected_message}")
        print(check_correction(encoded_message, corrected_message))

    if r == 4:
        codeword_with_3_errors = introduce_errors(encoded_message, [0, 1, 2])
        print(f"Сообщение с 3 ошибками: {codeword_with_3_errors}")
        corrected_message = correct_errors(codeword_with_3_errors, H, syndrome_table)
        print(f"Исправленное сообщение: {corrected_message}")
        print(check_correction(encoded_message, corrected_message))



def create_extended_H_matrix(r):
    H = create_H_matrix(r)

    vector_j = np.ones((H.shape[0], 1), dtype=int)
    H_ext = np.hstack((H, vector_j))

    zero_row = np.zeros((1, H_ext.shape[1]), dtype=int)
    zero_row[0, -1] = 1
    H_ext = np.vstack((H_ext, zero_row))

    return H_ext


def create_extended_G_matrix(H, r):
    G = create_G_matrix(H_ext[:-1, :-1], r)

    vector_b = (G.sum(axis=1) % 2).reshape(-1, 1)
    G_ext = np.hstack((G, vector_b))

    return G_ext


def create_extended_syndrome_table(H_ext):
    extended_syndrome_table = {}

    for i in range(H_ext.shape[0]):
        syndrome = H_ext[i, :]
        syndrome_str = np.array_str(syndrome)
        extended_syndrome_table[syndrome_str] = i

    return extended_syndrome_table


def extended_correct_errors(codeword, H_ext, syndrome_table):
    syndrome = compute_syndrome(codeword, H_ext)
    syndrome_str = np.array_str(syndrome)

    if syndrome_str in syndrome_table:
        error_position = syndrome_table[syndrome_str]
        print(f"Обнаружена ошибка на позиции: {error_position}")
        codeword[error_position] ^= 1  #исправляем ошибку
    else:
        print("Обнаружена двойная ошибка")

    return codeword


print("\nДля расширенного кода Хеммминга")


for r in [2, 3, 4]:
    print(f"\nИсследование для r = {r}")

    H_ext = create_extended_H_matrix(r)
    G_ext = create_extended_G_matrix(H_ext, r)
    extended_syndrome_table = create_extended_syndrome_table(H_ext)

    print("H (расширенная):\n", H_ext)
    print("G (расширенная):\n", G_ext)
    print("Таблица синдромов:\n", extended_syndrome_table)

    k = 2**r - r - 1
    message = np.random.randint(0, 2, k)
    print(f"Сообщение: {message}")

    encoded_message = encode_message(message, G_ext)
    print(f"Закодированное сообщение: {encoded_message}")

    codeword_with_1_error = introduce_errors(encoded_message, [0])
    print(f"Сообщение с 1 ошибкой: {codeword_with_1_error}")
    corrected_message = extended_correct_errors(codeword_with_1_error, H_ext, extended_syndrome_table)
    print(f"Исправленное сообщение: {corrected_message}")
    print(check_correction(encoded_message, corrected_message))

    if r > 2:
        codeword_with_2_errors = introduce_errors(encoded_message, [0, 1])
        print(f"Сообщение с 2 ошибками: {codeword_with_2_errors}")
        corrected_message = extended_correct_errors(codeword_with_2_errors, H_ext, extended_syndrome_table)
        print(f"Исправленное сообщение: {corrected_message}")
        print(check_correction(encoded_message, corrected_message))

        codeword_with_3_errors = introduce_errors(encoded_message, [0, 1, 2])
        print(f"Сообщение с 3 ошибками: {codeword_with_3_errors}")
        corrected_message = extended_correct_errors(codeword_with_3_errors, H_ext, extended_syndrome_table)
        print(f"Исправленное сообщение: {corrected_message}")
        print(check_correction(encoded_message, corrected_message))

    if r == 4:
        codeword_with_4_errors = introduce_errors(encoded_message, [0, 1, 2, 3])
        print(f"Сообщение с 4 ошибками: {codeword_with_4_errors}")
        corrected_message = extended_correct_errors(codeword_with_4_errors, H_ext, extended_syndrome_table)
        print(f"Исправленное сообщение: {corrected_message}")
        print(check_correction(encoded_message, corrected_message))
