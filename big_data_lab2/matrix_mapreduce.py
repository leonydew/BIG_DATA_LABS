from collections import defaultdict

#  MAPPER
def mapper_matrix(file_path, n):
    """
    Читает файл матрицы и формирует пары ключ-значение
    key = (i, j)
    value = ("A"/"B", k, value)
    """
    mapped = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            matrix, i, j, value = line.strip().split(",")
            i, j = int(i), int(j)
            value = float(value)

            if matrix == "A":
                # A[i][k] участвует во всех C[i][j]
                for col in range(n):
                    mapped.append(((i, col), ("A", j, value)))

            elif matrix == "B":
                # B[k][j] участвует во всех C[i][j]
                for row in range(n):
                    mapped.append(((row, j), ("B", i, value)))

    return mapped


# REDUCER 
def reducer_matrix(mapped):
    """
    Группирует значения по ключу (i, j) и считает сумму произведений
    """
    grouped = defaultdict(list)

    for key, value in mapped:
        grouped[key].append(value)

    result = {}

    for key, values in grouped.items():
        A_vals = {}
        B_vals = {}
        total = 0

        for tag, k, val in values:
            if tag == "A":
                A_vals[k] = val
            elif tag == "B":
                B_vals[k] = val

        for k in A_vals:
            if k in B_vals:
                total += A_vals[k] * B_vals[k]

        result[key] = total

    return result



def main():
    n = 2  # размерность матриц

    mapped = []
    mapped += mapper_matrix("matrix_A.txt", n)
    mapped += mapper_matrix("matrix_B.txt", n)

    result = reducer_matrix(mapped)

    print("Result matrix C = A x B:")
    for key in sorted(result):
        print(f"C{key} = {result[key]}")


if __name__ == "__main__":
    main()
