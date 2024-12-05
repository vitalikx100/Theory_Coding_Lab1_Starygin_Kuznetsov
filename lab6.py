import numpy as np
import random

"""Функция кодирования сообщения"""
def encode_message(input_message, generating_polynom):
    return np.polymul(input_message, generating_polynom) % 2

"""Функция для внесения ошибок в закодированное сообщение"""
def introduce_errors(codeword, error_count, is_packet_error=False):
    codeword_length = len(codeword)
    if is_packet_error:
        #Генерация пакета ошибок
        start_pos = random.randint(0, codeword_length - error_count)
        for i in range(error_count):
            codeword[(start_pos + i) % codeword_length] ^= 1
        print("Пакет ошибок внесён")
    else:
        #Генерация одиночных ошибок
        error_positions = random.sample(range(codeword_length), error_count)
        for position in error_positions:
            codeword[position] ^= 1
    return codeword


def is_single_error(error_pattern, max_errors):
    return sum(error_pattern) <= max_errors


def is_packet_error(error_pattern, max_errors):
    error_pattern = np.trim_zeros(error_pattern)
    return len(error_pattern) <= max_errors and len(error_pattern) != 0


def decode_message(received_word, generator_matrix, max_errors, error_check_function):
    syndrome = np.polydiv(received_word, generator_matrix)[1] % 2

    for i in range(len(received_word)):
        error_vector = np.zeros(len(received_word), dtype=int)
        error_vector[len(received_word) - i - 1] = 1
        result = np.polymul(syndrome, error_vector) % 2
        syndrome_at_pos = np.polydiv(result, generator_matrix)[1] % 2

        if error_check_function(syndrome_at_pos, max_errors):
            correction_vector = np.zeros(len(received_word), dtype=int)
            correction_vector[i - 1] = 1
            error_vector = np.polymul(correction_vector, syndrome_at_pos) % 2
            corrected_message = np.polyadd(error_vector, received_word) % 2
            decoded_message = np.array(np.polydiv(corrected_message, generator_matrix)[0] % 2).astype(int)
            return decoded_message
    return None

"""Универсальная функция для тестирования кодирования, внесения ошибок и декодирования"""
def test_code_correction(message, generator_matrix, max_errors, error_type, max_error_count):
    print("\nТестирование кода\n")
    for error_count in range(1, max_error_count + 1):
        print(f"Исходное сообщение: {message}")
        encoded_message = encode_message(message, generator_matrix)
        print(f"Закодированное сообщение: {encoded_message}")

        codeword_with_errors = introduce_errors(encoded_message.copy(), error_count, is_packet_error=(error_type == "packet"))
        print(f"Сообщение с ошибками: {codeword_with_errors}")

        #Выбор функции проверки ошибок в зависимости от типа ошибок
        error_check_function = is_packet_error if error_type == "packet" else is_single_error
        decoded_message = decode_message(codeword_with_errors, generator_matrix, max_errors, error_check_function)

        print(f"Декодированное сообщение: {decoded_message}")

        if np.array_equal(message, decoded_message):
            print("Сообщения совпадают.\n")
        else: print("Сообщения не совпадают.\n")

"""Тестирование кода (7,4)"""
test_code_correction(
    message=np.array([1, 0, 1, 0]),
    generator_matrix=np.array([1, 1, 0, 1]),
    max_errors=1,
    error_type="single",
    max_error_count=3
)

"""Тестирование кода (15,9) с пакетом ошибок"""
test_code_correction(
    message=np.array([1, 1, 0, 0, 0, 1, 0, 0, 0]),
    generator_matrix=np.array([1, 0, 0, 1, 1, 1, 1]),
    max_errors=3,
    error_type="packet",
    max_error_count=4
)
