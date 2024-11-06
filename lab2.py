import numpy as np
from itertools import combinations

# 2.1 Сформировать порождающую матрицу G для кода (7, 4, 3)
I_k = np.eye(4, dtype=int)

X = np.array([
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
])

G = np.hstack((I_k, X))
print("Порождающая матрица G:\n", G)

# 2.2 Сформировать проверочную матрицу H
I_nk = np.eye(3, dtype=int)
H = np.hstack((X.T, I_nk))
print("Проверочная матрица H:\n", H)

# 2.3 Сформировать таблицу синдромов для всех однократных ошибок
syndromes = {}
for i in range(7):
    # Создаем вектор ошибки с одной единицей
    error = np.zeros(7, dtype=int)
    error[i] = 1
    # Вычисляем синдром и преобразуем его к типу int
    syndrome = tuple(map(int, np.dot(H, error) % 2))
    syndromes[syndrome] = error


print("\nТаблица синдромов:")
for syndrome, error in syndromes.items():
    print(f"Синдром: {syndrome} => Ошибка: {error}")

# 2.4 Сформировать кодовое слово, внести однократную ошибку, вычислить синдром и исправить ошибку
def encode(message):
    return np.dot(message, G) % 2

def decode(received):
    syndrome = np.dot(H, received) % 2
    return tuple(map(int, syndrome))

# Слово длины k
message = np.array([1, 0, 1, 1])

# Кодируем сообщение
codeword = encode(message)
print("\nКодовое слово:", codeword)

# Вносим однократную ошибку

received = codeword.copy()
received[2] ^= 1  # Инвертируем бит для ошибки
print("Принятое слово с ошибкой:", received)

# Вычисляем синдром и исправляем ошибку
syndrome = decode(received)
print("Синдром:", syndrome)

if syndrome in syndromes:
    error = syndromes[syndrome]
    corrected = (received + error) % 2
    print("Исправленное слово:", corrected)
else:
    print("Ошибок не найдено или ошибка не подлежит исправлению.")

# Проверяем, что исправленное слово совпадает с оригинальным
if np.array_equal(corrected, codeword):
    print("Ошибка успешно исправлена.")
else:
    print("Ошибка не исправлена.")

# 2.5 Внести двукратную ошибку и проверить, что исправление невозможно
# Вносим двукратную ошибку
received = codeword.copy()
received[1] ^= 1  # Инвертируем первый бит
received[4] ^= 1  # Инвертируем другой бит для двукратной ошибки
print("\nПринятое слово с двукратной ошибкой:", received)

# Вычисляем синдром для двукратной ошибки
syndrome = decode(received)
print("Синдром для двукратной ошибки:", syndrome)

# Пытаемся исправить ошибку с использованием таблицы синдромов
if syndrome in syndromes:
    error = syndromes[syndrome]
    corrected = (received + error) % 2
    print("Исправленное слово:", corrected)
else:
    print("Ошибок не найдено или ошибка не подлежит исправлению.")

# Проверяем, что исправленное слово совпадает с оригинальным
if np.array_equal(corrected, codeword):
    print("Ошибка успешно исправлена.")
else:
    print("Ошибка не исправлена.")

# 2.6 Сформировать порождающую матрицу G для кода (n, k, 5)

X = np.array([
    [1, 1, 1, 1, 0, 0, 0],
    [1, 1, 0, 0, 1, 1, 0],
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 1]
])


I_k = np.eye(5, dtype=int)
G = np.hstack((I_k, X))

print("Порождающая матрица G для кода (n, k, 5):\n", G)

# 2.7 Сформировать проверочную матрицу H

I_nk = np.eye(7, dtype=int)
H = np.hstack((X.T, I_nk))
print("Проверочная матрица H для кода (n, k, 5):\n", H)

# 2.8 Сформировать таблицу синдромов для всех однократных и двукратных ошибок

syndromes = {}

# Однократные ошибки
for i in range(12):
    error = np.zeros(12, dtype=int)
    error[i] = 1
    syndrome = tuple(map(int, np.dot(H, error) % 2))
    syndromes[syndrome] = error

# Двукратные ошибки
for i, j in combinations(range(12), 2):
    error = np.zeros(12, dtype=int)
    error[i] = 1
    error[j] = 1
    syndrome = tuple(map(int, np.dot(H, error) % 2))
    if syndrome not in syndromes:
        syndromes[syndrome] = error

print("\nТаблица синдромов (однократные и двукратные ошибки):")
for syndrome, error in syndromes.items():
    print(f"Синдром: {syndrome} => Ошибка: {error}")

# Функции кодирования и декодирования
def encode(message):
    return np.dot(message, G) % 2

def decode(received):
    syndrome = np.dot(H, received) % 2
    return tuple(map(int, syndrome))

# 2.9 Сформировать кодовое слово длины n из слова длины k, внести однократную ошибку, исправить
message = np.array([1, 0, 1, 0, 1])
codeword = encode(message)
print("\nКодовое слово:", codeword)

# Вносим однократную ошибку
received = codeword.copy()
received[3] ^= 1  # Инвертируем бит
print("Принятое слово с однократной ошибкой:", received)

# Исправляем ошибку
syndrome = decode(received)
print("Синдром:", syndrome)
if syndrome in syndromes:
    error = syndromes[syndrome]
    corrected = (received + error) % 2
    print("Исправленное слово:", corrected)

# 2.10 Внести двукратную ошибку, исправить и проверить результат
received = codeword.copy()
received[2] ^= 1  # Первая ошибка
received[6] ^= 1  # Вторая ошибка
print("\nПринятое слово с двукратной ошибкой:", received)

# Исправляем двукратную ошибку
syndrome = decode(received)
print("Синдром для двукратной ошибки:", syndrome)
if syndrome in syndromes:
    error = syndromes[syndrome]
    corrected = (received + error) % 2
    print("Исправленное слово:", corrected)

# 2.11 Внести трехкратную ошибку и проверить, что код не может ее исправить
received = codeword.copy()
received[1] ^= 1  # Первая ошибка
received[4] ^= 1  # Вторая ошибка
received[8] ^= 1  # Третья ошибка
print("\nПринятое слово с трехкратной ошибкой:", received)

# Пытаемся исправить трехкратную ошибку
syndrome = decode(received)
print("Синдром для трехкратной ошибки:", syndrome)
if syndrome in syndromes:
    error = syndromes[syndrome]
    corrected = (received + error) % 2
    print("Исправленное слово:", corrected)
else:
    print("Ошибка не подлежит исправлению")