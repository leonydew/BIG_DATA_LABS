import ast
import sys

import warnings
warnings.simplefilter("ignore", SyntaxWarning)


# ------------------- MAPPER -------------------
def mapper(input_file, intermediate_file):
    """
    Читает train.csv, где каждая строка — список ['score', 'summary', 'text'].
    Пишет пары summary<TAB>score в промежуточный файл.
    """
    with open(input_file, "r", encoding="utf-8") as f, \
         open(intermediate_file, "w", encoding="utf-8") as out:

        for line in f:
            try:
                row = ast.literal_eval(line.strip())
                if len(row) >= 2:
                    score = float(row[0])
                    summary = str(row[1]).replace("\n", " ").replace("\r", " ").replace("\t", " ")
                    out.write(f"{summary}\t{score}\n")
            except:
                continue

    print(f"Mapper finished: {intermediate_file} created.")


# ------------------- STREAMING REDUCER -------------------
def reducer_streaming(sorted_file, output_file):
    """
    Потоковый reducer.
    Считает количество отзывов и средний рейтинг для каждой summary.
    Пропускает некорректные строки.
    """
    with open(sorted_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        current_summary = None
        count = 0
        sum_ratings = 0.0

        for line in fin:
            line = line.strip()
            if not line or "\t" not in line:
                continue  # пропускаем некорректные строки

            # Разделяем только по первому табу, чтобы табы в summary не ломали reducer
            summary, rating_str = line.split("\t", 1)
            try:
                rating = float(rating_str)
            except:
                continue

            if current_summary is None:
                current_summary = summary

            if summary == current_summary:
                count += 1
                sum_ratings += rating
            else:
                avg = sum_ratings / count
                fout.write(f"{current_summary}\t{count}\t{avg:.4f}\n")
                current_summary = summary
                count = 1
                sum_ratings = rating

        # последний ключ
        if current_summary is not None and count > 0:
            avg = sum_ratings / count
            fout.write(f"{current_summary}\t{count}\t{avg:.4f}\n")

    print(f"Reducer finished: {output_file} created.")


# ------------------- RANKING -------------------
def ranking(output_file):
    """
    Вывод топ-50 summary:
    - по числу отзывов
    - по средней оценке (мин 100 отзывов)
    """
    items = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            summary, count, avg = parts
            items.append((summary, int(count), float(avg)))

    top_count = sorted(items, key=lambda x: x[1], reverse=True)[:50]
    filtered = [x for x in items if x[1] >= 100]
    top_rating = sorted(filtered, key=lambda x: x[2], reverse=True)[:50]
    return top_count, top_rating


# ------------------- MAIN -------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python mapreduce_amazon.py train.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    intermediate_file = "mapped.txt"
    sorted_file = "mapped_sorted.txt"
    reduced_file = "reduced.txt"

    # 1. Mapper
    print("Running mapper...")
    mapper(input_file, intermediate_file)

    # 2. Сортировка по summary
    print("Sorting intermediate file...")
    with open(intermediate_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines.sort(key=lambda x: x.split("\t")[0])
    with open(sorted_file, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Sorted file created: {sorted_file}")

    # 3. Reducer
    print("Running reducer...")
    reducer_streaming(sorted_file, reduced_file)

    # 4. Ranking
    print("Calculating top summaries...")
    top_count, top_rating = ranking(reduced_file)

    print("\nTOP-50 by review count:")
    for summary, c, a in top_count:
        print(summary, c, a)

    print("\nTOP-50 by average score (min 100 reviews):")
    for summary, c, a in top_rating:
        print(summary, c, a)


if __name__ == "__main__":
    main()
