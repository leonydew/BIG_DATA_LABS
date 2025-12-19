# MAPPER
def mapper_lr(file_path):
    """
    Для каждой точки (x, y) формирует статистики
    """
    mapped = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            x, y = map(float, line.strip().split(","))

            mapped.append(("sum_x", x))
            mapped.append(("sum_y", y))
            mapped.append(("sum_xy", x * y))
            mapped.append(("sum_x2", x * x))
            mapped.append(("count", 1))

    return mapped


# REDUCER
def reducer_lr(mapped):
    """
    Агрегирует суммы и считает коэффициенты линейной регрессии
    """
    sums = {}

    for key, value in mapped:
        sums[key] = sums.get(key, 0) + value

    n = sums["count"]
    sum_x = sums["sum_x"]
    sum_y = sums["sum_y"]
    sum_xy = sums["sum_xy"]
    sum_x2 = sums["sum_x2"]

    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_y - a * sum_x) / n

    return a, b



def main():
    mapped = mapper_lr("points.txt")
    a, b = reducer_lr(mapped)

    print(f"Linear regression equation:")
    print(f"y = {a:.4f}x + {b:.4f}")


if __name__ == "__main__":
    main()
