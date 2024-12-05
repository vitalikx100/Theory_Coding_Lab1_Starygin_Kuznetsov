import numpy as np
from itertools import combinations, product
from statistics import mode

def generate_binary_vectors(num_cols):
    """
    Генерирует все возможные бинарные векторы длины num_cols.
    """
    vectors = product([0, 1], repeat=num_cols)
    binary_vectors = []
    for vector in vectors:
        binary_vectors.append(list(vector)[::-1])
    return binary_vectors

def check_vector(indexes, u):
    """
    Проверяет, равны ли значения вектора u по индексам в indexes нулю.
    Возвращает 1, если равны, иначе 0.
    """
    if len(indexes) == 0:
        return 1
    else:
        u_array = np.asarray(u)
        if np.all(u_array[indexes] == 0):
            return 1
        else:
            return 0

def vectorize_function(indexes, size):
    """
    Преобразует все бинарные векторы длины size и проверяет их с помощью check_vector.
    """
    result = []
    binary_vectors = generate_binary_vectors(size)
    for b in binary_vectors:
        result.append(check_vector(indexes, b))
    return result

def reed_muller_G(r, m):
    """
    Генерирует порождающую матрицу Гильберта для кода Рида-Маллера.
    """
    index_combinations = generate_all_indexes(r, m)
    G_matrix = []
    for idx in index_combinations:
        G_matrix.append(vectorize_function(idx, m))
    return np.array(G_matrix)

def reed_muller_H(index, m):
    """
    Генерирует проверочную матрицу для индекса в коде Рида-Маллера.
    """
    H_matrix = []
    binary_vectors = generate_binary_vectors(m)
    for u in binary_vectors:
        if check_vector(index, u) == 1:
            H_matrix.append(u)
    return H_matrix

def get_complementary_indices(indexes, m):
    """
    Получает индексы, которые не входят в набор indexes в пределах от 0 до m-1.
    """
    complementary_indices = []
    for i in range(m):
        if i not in indexes:
            complementary_indices.append(i)
    return complementary_indices

def get_combinations(size, m):
    """
    Генерирует все комбинации индексов размером size из диапазона от 0 до m-1.
    """
    combinations_list = []
    for comb in combinations(range(m - 1, -1, -1), size):
        combinations_list.append(list(comb))
    return combinations_list

def generate_all_indexes(r, m):
    """
    Генерирует все возможные индексы для кодов Рида-Маллера, для всех степеней от 0 до r.
    """
    index_array = []
    for i in range(r + 1):
        combinations_list = get_combinations(i, m)
        index_array.extend(combinations_list)
    return index_array

def f_with_t(index, t, m):
    """
    Проверяет на совпадение бинарные векторы для индексов и вектора t.
    """
    result = []
    binary_vectors = generate_binary_vectors(m)
    for b in binary_vectors:
        result.append(int(np.array_equal(np.asarray(b)[index], np.asarray(t)[index])))
    return result

def majority_vote_decoding(word_with_error, H, idx, m):
    """
    Декодирует слово с ошибкой с помощью голосования по большинству для матрицы H.
    """
    complementary = get_complementary_indices(idx, m)
    vote_vectors = []
    for u in H:
        vote_vectors.append(f_with_t(complementary, u, m))
    votes = []
    for v in vote_vectors:
        votes.append(np.dot(np.asarray(v), np.asarray(word_with_error)) % 2)
    return mode(votes)

def decode_with_reed_muller(word_with_error, r, m, G):
    """
    Декодирует слово с ошибкой, используя алгоритм Рида-Маллера.
    """
    decoded_word = np.zeros((G.shape[0]), dtype=int)
    all_indexes = generate_all_indexes(r, m)

    for step in range(r, -1, -1):
        index_combinations = get_combinations(step, m)
        first_step_results = []

        for idx in index_combinations:
            H = reed_muller_H(idx, m)
            first_step_results.append(majority_vote_decoding(word_with_error, H, idx, m))

        position = all_indexes.index(index_combinations[0])

        for i in range(len(first_step_results)):
            decoded_word[i + position] = first_step_results[i]


        if step != 0:
            word_with_error = (decoded_word.T @ G + word_with_error) % 2
        else:
            word_with_error = decoded_word.T @ G % 2

        print(f'Слово после шага {abs(step - r - 1)}: {word_with_error}')

    return word_with_error

def reed_muller_experiment():
    """
    Эксперимент с кодом Рида-Маллера: генерация порождающей матрицы, шифрование, добавление ошибок и декодирование.
    """
    r, m = 2, 4
    G = reed_muller_G(r, m)
    print('Порождающая матрица G: \n', G)

    original_word = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    print('Исходное слово: ', original_word)

    encrypted_word = original_word @ G % 2
    print('Зашифрованное слово: ', encrypted_word)

    E = np.eye(16, dtype=int)
    word_with_error = (encrypted_word + E[4]) % 2
    print('Слово с ошибкой: ', word_with_error)

    decoded_word = decode_with_reed_muller(word_with_error, r, m, G)
    print('Исправленное сообщение: ', decoded_word)

    if np.array_equal(encrypted_word, decoded_word):
        print("Отправленное слово и декодированное совпадают.\n")
    else:
        print("Отправленное слово и декодированное не совпадают.\n")

# Запуск эксперимента
reed_muller_experiment()
