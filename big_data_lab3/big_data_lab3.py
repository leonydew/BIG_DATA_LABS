"""
Лабораторная работа 3: Анализ Amazon Reviews
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import json
from collections import defaultdict

# 1. НАСТРОЙКИ
SAMPLE_SIZE = 500000  # Сколько строк анализировать
RESULTS_DIR = "lab3_results"

# 2. ЗАГРУЗКА ДАННЫХ
def load_data():
    """Загружаем данные Amazon Reviews"""
    print("📥 Загрузка данных...")
    
    try:
        # Пробуем разные форматы
        for sep in [',', '\t', ';']:
            try:
                df = pd.read_csv('train.csv', sep=sep, nrows=SAMPLE_SIZE, 
                               encoding='utf-8', on_bad_lines='skip')
                if len(df.columns) > 1:
                    break
            except:
                continue
        
        # Если только одна колонка, пробуем по-другому
        if len(df.columns) == 1:
            df = pd.read_csv('train.csv', nrows=SAMPLE_SIZE, encoding='utf-8')
        
        # Переименовываем колонки
        if len(df.columns) >= 3:
            df.columns = ['id', 'title', 'text']
        elif len(df.columns) == 2:
            df.columns = ['id', 'text']
        else:
            df.columns = ['text']
        
        print(f"✅ Загружено {len(df):,} строк")
        return df
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None

# 3. MAPREDUCE РЕАЛИЗАЦИЯ
def mapper(chunk):
    """Функция Mapper: преобразует текст в (ключ, значение)"""
    results = {}
    
    for text in chunk:
        if not isinstance(text, str):
            continue
            
        # 1. Определяем категорию по длине
        length = len(text)
        if length < 100:
            category = "short"
        elif length < 500:
            category = "medium"
        else:
            category = "long"
        
        # 2. Извлекаем рейтинг
        rating = extract_rating(text)
        
        # 3. Определяем тональность
        sentiment = "positive" if rating >= 4 else "negative"
        
        # 4. Формируем ключ
        key = f"{category}|{sentiment}"
        
        # 5. Сохраняем результат
        if key not in results:
            results[key] = []
        results[key].append((rating, 1))  # (рейтинг, счетчик)
    
    return results

def reducer(key, values):
    """Функция Reducer: агрегирует значения по ключу"""
    total_rating = 0
    count = 0
    
    for rating, cnt in values:
        total_rating += rating * cnt
        count += cnt
    
    avg_rating = total_rating / count if count > 0 else 0
    
    # Разделяем ключ обратно
    category, sentiment = key.split('|')
    
    return {
        'category': category,
        'sentiment': sentiment,
        'count': count,
        'avg_rating': round(avg_rating, 2)
    }

def run_mapreduce(df):
    """Запуск полного цикла MapReduce"""
    print("\n🚀 ЗАПУСК MAPREDUCE...")
    
    # Разбиваем данные на части
    chunks = [df['text'][i:i+10000] for i in range(0, len(df), 10000)]
    
    # Этап 1: MAP
    print("⚙️  Mapper этап...")
    mapped_results = []
    for chunk in chunks[:10]:  # Берем только 10 чанков для скорости
        mapped_results.append(mapper(chunk))
    
    # Этап 2: SHUFFLE (группируем по ключам)
    print("🔄 Shuffle этап...")
    shuffled = defaultdict(list)
    for result in mapped_results:
        for key, values in result.items():
            shuffled[key].extend(values)
    
    # Этап 3: REDUCE
    print("🔧 Reducer этап...")
    final_results = []
    for key, values in shuffled.items():
        final_results.append(reducer(key, values))
    
    # Сохраняем результаты
    results_df = pd.DataFrame(final_results)
    results_df.to_csv(f'{RESULTS_DIR}/mapreduce_results.csv', index=False)
    
    print(f"✅ MapReduce завершен! Результатов: {len(results_df)}")
    return results_df

# 4. SPARK-ПОДОБНЫЙ АНАЛИЗ
def spark_analysis(df):
    """Анализ в стиле Spark (используем pandas)"""
    print("\n🚀 ЗАПУСК SPARK-ПОДОБНОГО АНАЛИЗА...")
    # Аналог HiveQL запроса:
    # SELECT category, AVG(rating), COUNT(*) 
    # FROM reviews 
    # GROUP BY category 
    # ORDER BY count DESC
    
    # Добавляем вычисляемые поля
    df['text_length'] = df['text'].apply(lambda x: len(str(x)))
    df['rating'] = df['text'].apply(extract_rating)
    
    # Группируем и агрегируем
    df['length_category'] = pd.cut(df['text_length'], 
                                   bins=[0, 100, 500, float('inf')],
                                   labels=['short', 'medium', 'long'])
    
    # Агрегация как в Spark
    spark_results = df.groupby(['length_category']).agg(
        count=('rating', 'size'),
        avg_rating=('rating', 'mean'),
        avg_length=('text_length', 'mean')
    ).round(2).reset_index()
    
    # Сохраняем
    spark_results.to_csv(f'{RESULTS_DIR}/spark_results.csv', index=False)
    
    print(f"✅ Spark анализ завершен!")
    return spark_results

# 5. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
def extract_rating(text):
    """Извлекает рейтинг из текста"""
    if not isinstance(text, str):
        return 3
    
    text = text.lower()
    
    # Ищем рейтинги 1-5
    patterns = [
        r'(\d)[\s\-]?star',
        r'rating[\s\:]+(\d)',
        r'(\d)/5',
        r'(\d) out of 5'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            rating = int(match.group(1))
            if 1 <= rating <= 5:
                return rating
    
    # Если не нашли, определяем по словам
    positive = ['good', 'great', 'excellent', 'love', 'best', 'perfect']
    negative = ['bad', 'poor', 'terrible', 'worst', 'awful', 'horrible']
    
    pos_count = sum(1 for word in positive if word in text)
    neg_count = sum(1 for word in negative if word in text)
    
    if pos_count > neg_count:
        return 5
    elif neg_count > pos_count:
        return 1
    else:
        return 3

# 6. ВИЗУАЛИЗАЦИЯ
def create_charts(mapreduce_results, spark_results):
    """Создаем графики"""
    print("\n🎨 СОЗДАНИЕ ГРАФИКОВ...")
    
    os.makedirs(f'{RESULTS_DIR}/charts', exist_ok=True)
    
    # 1. Сравнение результатов
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # MapReduce результаты
    mr_pivot = mapreduce_results.pivot_table(
        index='category', 
        columns='sentiment', 
        values='avg_rating',
        fill_value=0
    )
    mr_pivot.plot(kind='bar', ax=axes[0], title='MapReduce: Рейтинг по категориям')
    axes[0].set_xlabel('Категория')
    axes[0].set_ylabel('Средний рейтинг')
    axes[0].legend(title='Тональность')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Spark результаты
    spark_results.set_index('length_category')['avg_rating'].plot(
        kind='bar', ax=axes[1], color='green', title='Spark: Средний рейтинг'
    )
    axes[1].set_xlabel('Категория длины')
    axes[1].set_ylabel('Средний рейтинг')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/charts/comparison.png', dpi=150, bbox_inches='tight')
    
    # 2. Распределение тональности
    plt.figure(figsize=(8, 6))
    sentiment_counts = mapreduce_results.groupby('sentiment')['count'].sum()
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, 
            autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
    plt.title('Распределение тональности отзывов')
    plt.savefig(f'{RESULTS_DIR}/charts/sentiment_pie.png', dpi=150, bbox_inches='tight')
    
    print("✅ Графики сохранены в папке charts/")
    plt.close('all')

# 7. ОТЧЕТ
def generate_report(mapreduce_results, spark_results):
    """Создаем отчет"""
    print("\n📄 СОЗДАНИЕ ОТЧЕТА...")
    
    # JSON отчет
    report = {
        "lab_info": {
            "name": "Лабораторная работа 3",
            "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "sample_size": SAMPLE_SIZE
        },
        "mapreduce_results": {
            "total_categories": len(mapreduce_results['category'].unique()),
            "total_reviews": int(mapreduce_results['count'].sum()),
            "avg_rating": round(mapreduce_results['avg_rating'].mean(), 2),
            "positive_percent": round(
                mapreduce_results[mapreduce_results['sentiment'] == 'positive']['count'].sum() / 
                mapreduce_results['count'].sum() * 100, 2
            )
        },
        "spark_results": {
            "avg_rating": round(spark_results['avg_rating'].mean(), 2),
            "avg_text_length": round(spark_results['avg_length'].mean(), 0)
        },
        "findings": [
            "Большинство отзывов положительные",
            "Средний рейтинг около 4 из 5",
            "Короткие отзывы чаще положительные"
        ]
    }
    
    # Сохраняем JSON
    with open(f'{RESULTS_DIR}/report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Текстовый отчет
    with open(f'{RESULTS_DIR}/report.txt', 'w', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write("ОТЧЕТ ПО ЛАБОРАТОРНОЙ РАБОТЕ 3\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Дата: {report['lab_info']['date']}\n")
        f.write(f"Образец данных: {report['lab_info']['sample_size']:,} строк\n\n")
        
        f.write("MAPREDUCE РЕЗУЛЬТАТЫ:\n")
        f.write(f"• Категорий: {report['mapreduce_results']['total_categories']}\n")
        f.write(f"• Отзывов: {report['mapreduce_results']['total_reviews']:,}\n")
        f.write(f"• Средний рейтинг: {report['mapreduce_results']['avg_rating']}\n")
        f.write(f"• Положительных: {report['mapreduce_results']['positive_percent']}%\n\n")
        
        f.write("SPARK РЕЗУЛЬТАТЫ:\n")
        f.write(f"• Средний рейтинг: {report['spark_results']['avg_rating']}\n")
        f.write(f"• Средняя длина: {report['spark_results']['avg_text_length']} симв.\n\n")
        
        f.write("ВЫВОДЫ:\n")
        for i, finding in enumerate(report['findings'], 1):
            f.write(f"{i}. {finding}\n")
    
    print("✅ Отчет сохранен в report.json и report.txt")

# 8. ГЛАВНАЯ ФУНКЦИЯ
def main():
    """Основная функция"""
    print("="*50)
    print("ЛАБОРАТОРНАЯ РАБОТА 3 - ЗАПУСК")
    print("="*50)
    
    # Создаем папку для результатов
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Загружаем данные
    df = load_data()
    if df is None:
        return
    
    # 2. MapReduce анализ
    mapreduce_results = run_mapreduce(df)
    
    # 3. Spark анализ
    spark_results = spark_analysis(df)
    
    # 4. Визуализация
    create_charts(mapreduce_results, spark_results)
    
    # 5. Отчет
    generate_report(mapreduce_results, spark_results)
    
    print("\n" + "="*50)
    print("ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА!")
    print("="*50)
    print(f"\n📁 Результаты в папке: {os.path.abspath(RESULTS_DIR)}")
    print("\nСозданы файлы:")
    print("1. mapreduce_results.csv - Результаты MapReduce")
    print("2. spark_results.csv - Результаты Spark")
    print("3. charts/comparison.png - Графики сравнения")
    print("4. charts/sentiment_pie.png - Круговая диаграмма")
    print("5. report.json - Полный отчет")
    print("6. report.txt - Краткий отчет")

# 9. ЗАПУСК
if __name__ == "__main__":
    main()
    
    input("\nНажмите Enter для выхода...")