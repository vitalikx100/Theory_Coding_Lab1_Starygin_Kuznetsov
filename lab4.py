import numpy as np
from itertools import combinations
import random


def generate_golay_matrices():
    # Заданная матрица B для кода Голея (24,12,8)
    B = np.array([
        [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
        [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
        [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    ])
    I = np.identity(12, dtype=int)
    G = np.hstack((I, B))  # Порождающая матрица

    # Проверочная матрица для кода Голея (24,12,8)
    H = np.vstack((I, B))  # Проверочная матрица H = [I; B]
    return G, H, B


def introduce_errors(codeword, num_errors):
    """Вносит заданное количество ошибок в случайные позиции кодового слова."""
    error_positions = random.sample(range(24), num_errors)
    corrupted_codeword = codeword.copy()
    for pos in error_positions:
        corrupted_codeword[pos] ^= 1  # Инвертируем бит
    return corrupted_codeword, error_positions


def hamming_weight(vector):
    """Вычисляет вес Хэмминга (количество единиц) в векторе."""
    return np.sum(vector)


def decode_golay(codeword, H, B):
    """Декодирует кодовое слово, исправляя до 3 ошибок."""
    print(f"Исходное кодовое слово: {codeword}")

    # Шаг 1: Вычисляем синдром s = wH
    syndrome = (codeword @ H) % 2
    print(f"Синдром s: {syndrome[:12]} (первая часть), {syndrome[12:]} (вторая часть)")

    # Шаг 2: Если wt(s) <= 3, исправляем информационную часть
    if hamming_weight(syndrome[:12]) <= 3:
        corrected_codeword = codeword.copy()
        corrected_codeword[:12] ^= syndrome[:12]
        print(f"Исправлено по шагу 2: {corrected_codeword}")
        return corrected_codeword, True

    # Шаг 3: Если wt(s + b_i) <= 2
    for i in range(12):
        test_syndrome = (syndrome[:12] + B[i]) % 2
        if hamming_weight(test_syndrome) <= 2:
            corrected_codeword = codeword.copy()
            corrected_codeword[12 + i] ^= 1  # инвертируем бит
            corrected_codeword[:12] ^= test_syndrome
            print(f"Исправлено по шагу 3: {corrected_codeword}")
            return corrected_codeword, True

    # Шаг 4: Вычисляем второй синдром sB
    syndrome_B = (syndrome[:12] @ B) % 2
    print(f"Второй синдром sB: {syndrome_B}")

    # Шаг 5: Если wt(sB) <= 3
    if hamming_weight(syndrome_B) <= 3:
        corrected_codeword = codeword.copy()
        corrected_codeword[12:] ^= syndrome_B
        print(f"Исправлено по шагу 5: {corrected_codeword}")
        return corrected_codeword, True

    # Шаг 6: Если wt(sB + b_i) <= 2
    for i in range(12):
        test_syndrome_B = (syndrome_B + B[i]) % 2
        if hamming_weight(test_syndrome_B) <= 2:
            corrected_codeword = codeword.copy()
            corrected_codeword[i] ^= 1  #инвертируем бит
            corrected_codeword[12:] ^= test_syndrome_B
            print(f"Исправлено по шагу 6: {corrected_codeword}")
            return corrected_codeword, True

    print("Ошибки не удалось исправить")
    return codeword, False  # Ошибки не удалось исправить


def investigate_golay_code_debug():
    G, H, B = generate_golay_matrices()
    print("Порождающая матрица (G):")
    print(G)
    print("\nПроверочная матрица (H):")
    print(H)

    error_tests = [1, 2, 3, 4]
    for num_errors in error_tests:
        print(f"\nПроверка исправления {num_errors}-кратных ошибок:")
        # Генерируем случайное сообщение
        message = np.random.randint(2, size=12)
        # Кодируем сообщение
        codeword = (message @ G) % 2
        # Вносим ошибки
        corrupted_codeword, error_positions = introduce_errors(codeword, num_errors)
        print(f"Позиции ошибок: {error_positions}")

        # Пытаемся исправить ошибки
        corrected_codeword, success = decode_golay(corrupted_codeword, H, B)

        # Проверка на успешное исправление
        if success and np.array_equal(corrected_codeword, codeword):
            print(f"{num_errors}-кратные ошибки успешно исправлены.")
        else:
            print(f"{num_errors}-кратные ошибки не удалось исправить.")


investigate_golay_code_debug()


def generate_rm_matrix(r, m):
    """
    Рекурсивно строит порождающую матрицу G(r, m) для кода Рида-Маллера.
    :param r: Параметр порядка кода (r).
    :param m: Параметр длины (m).
    :return: Порождающая матрица G(r, m).
    """
    if r == 0:
        # Для r = 0: матрица состоит из одной строки с единицами
        return np.ones((1, 2**m), dtype=int)
    elif r == m:
        # Для r = m: расширение G(m-1, m) с добавлением строки [0 ... 0 1]
        G_prev = generate_rm_matrix(m-1, m)
        extra_row = np.zeros((1, 2**m), dtype=int)
        extra_row[0, -1] = 1  # Последний элемент равен 1
        return np.vstack((G_prev, extra_row))
    else:
        # Рекурсивное определение
        G_rm1 = generate_rm_matrix(r, m - 1)  # G(r, m-1)
        G_r1m1 = generate_rm_matrix(r - 1, m - 1)  # G(r-1, m-1)

        # Верхняя часть: [G(r, m-1), G(r, m-1)]
        top = np.hstack((G_rm1, G_rm1))
        # Нижняя часть: [0, G(r-1, m-1)]
        bottom = np.hstack((np.zeros_like(G_r1m1), G_r1m1))

        # Объединяем верхнюю и нижнюю части
        return np.vstack((top, bottom))




def generate_h_matrix(i, m):
    """
    Генерирует проверочную матрицу H^i_m для заданных параметров m и i.
    :param m: Параметр длины кода Рида-Маллера.
    :param i: Индекс i (номер матрицы H^i_m).
    :return: Проверочная матрица H^i_m.
    """
    # Базовая матрица H
    H_base = np.array([[1, 1], [1, -1]])

    # Создаём единичные матрицы соответствующих размеров
    I_left = np.eye(2**(m - i), dtype=int)
    I_right = np.eye(2**(i - 1), dtype=int)

    # Вычисляем произведение Кронекера: H^i_m = I_left ⊗ H_base ⊗ I_right
    H_i_m = np.kron(np.kron(I_left, H_base), I_right)

    return H_i_m

print(generate_h_matrix(1, 3))


def decode_rm_1_m(received, m):
    """
    Быстрый алгоритм декодирования для RM(1, m).
    :param received: Принятое сообщение (вектор длины 2^m).
    :param m: Параметр длины кода Рида-Маллера.
    :return: Декодированное сообщение.
    """
    # Преобразуем вектор w: заменяем 0 на -1
    w = np.array([1 if bit == 1 else -1 for bit in received])
    print(f"Начальное w: {w}")

    #вычисление w_i = w_{i-1} * H^i_m
    for i in range(1, m + 1):
        H_i_m = generate_h_matrix(i, m)  # Получаем матрицу H^i_m
        w = np.dot(w, H_i_m)  # Умножаем w на H^i_m
        print(f"Шаг {i}, w = {w}")

    # Находим позицию j, соответствующую максимальному абсолютному значению компонента w
    j = np.argmax(np.abs(w))

    # Определяем декодированное сообщение
    v_j = [(j >> k) & 1 for k in range(m)]  # Двоичное представление позиции j
    v_j.reverse()  # Переворачиваем младшие биты в начало

    # Если компонент положительный, сообщение начинается с 1
    if w[j] > 0:
        decoded_message = [1] + v_j
    else:  # Если компонент отрицательный, сообщение начинается с 0
        decoded_message = [0] + v_j

    decoded_message = list(map(int, decoded_message))


    return decoded_message

def investigate_rm(r, m, max_errors):
    """
    Исследование кода Рида-Маллера RM(r, m) для различных ошибок.
    """
    G = generate_rm_matrix(r, m)
    print(f"=== Исследование кода RM({r}, {m}) ===")
    print(f"Порождающая матрица G:\n{G}\n")

    n = 2 ** m
    k = G.shape[0]

    if m == 3: message = np.array([1, 0, 1, 0])
    else: message = np.array([0, 0, 1, 1, 0])
    codeword = np.dot(message, G) % 2
    print(f"Сообщение: {message}")
    print(f"Кодовое слово: {codeword}\n")

    for num_errors in range(1, max_errors + 1):
        error_positions = list(combinations(range(n), num_errors))
        for positions in error_positions[:1]:  # Ограничимся одним примером для наглядности
            corrupted = codeword.copy()
            for pos in positions:
                corrupted[pos] ^= 1
            print(f"Вносим {num_errors} ошибок:")
            print(f"Искажённое кодовое слово: {corrupted}")
            print(f"Ошибки внесены в позиции: {positions}")

            try:
                decoded_message = decode_rm_1_m(corrupted, m)
                print(f"Декодированное сообщение: {decoded_message}\n")
            except Exception as e:
                print(f"Ошибка при декодировании: {e}\n")

# Проведение исследования для RM(1, 3) и RM(1, 4)
investigate_rm(1, 3, 2)  # RM(1,3) для одно- и двухкратных ошибок
investigate_rm(1, 4, 4)  # RM(1,4) для одно-, двух-, трёх- и четырёхкратных ошибок