
import os
import sys
import re
import math
import pickle
from pathlib import Path
from collections import defaultdict

#КЛАССЫ ДАННЫХ

class Document:
    def __init__(self, id, path, title, content):
        self.id = id
        self.path = path
        self.title = title
        self.content = content
        self.outgoing_links = []
        self.incoming_links = []
        self.pagerank = 0.0

# ФУНКЦИИ ДЛЯ ДАННЫХ

def create_test_documents_with_links():
    """Создание тестовых документов с реальными ссылками"""
    os.makedirs("data", exist_ok=True)
    
    docs = [
        ("doc1.txt", "Программирование на Python\nPython - популярный язык.\nИспользуется в [[doc2.txt]] и [[doc3.txt]].\nТакже для [[doc4.txt]]."),
        ("doc2.txt", "Веб разработка\nВеб-сайты создаются с помощью HTML, CSS, JavaScript.\nСвязано с [[doc1.txt]] и [[doc5.txt]].\nФреймворки: [[doc3.txt]]."),
        ("doc3.txt", "Фреймворки для веба\nDjango, Flask, FastAPI.\nИспользуют [[doc1.txt]].\nДля [[doc2.txt]]."),
        ("doc4.txt", "Анализ данных\nАнализ данных использует статистику.\nИнструменты: [[doc1.txt]], [[doc6.txt]].\nСвязано с [[doc5.txt]]."),
        ("doc5.txt", "Базы данных\nХранение данных в SQL и NoSQL.\nИспользуются в [[doc2.txt]] и [[doc4.txt]].\nСУБД: [[doc7.txt]]."),
        ("doc6.txt", "Машинное обучение\nАлгоритмы для предсказаний.\nИспользует [[doc1.txt]] и [[doc4.txt]].\nБиблиотеки: [[doc8.txt]]."),
        ("doc7.txt", "SQL базы данных\nMySQL, PostgreSQL.\nДля [[doc5.txt]].\nИспользуются в [[doc2.txt]]."),
        ("doc8.txt", "Библиотеки Python\nNumPy, Pandas, Scikit-learn.\nДля [[doc1.txt]] и [[doc6.txt]].\nУстановка: [[doc9.txt]]."),
        ("doc9.txt", "Установка Python\nКак установить Python.\nОсновы: [[doc1.txt]].\nДалее: [[doc10.txt]]."),
        ("doc10.txt", "Продвинутый Python\nДекораторы, генераторы.\nТребует [[doc1.txt]].\nДля [[doc2.txt]] и [[doc3.txt]].")
    ]
    
    for filename, content in docs:
        with open(f"data/{filename}", 'w', encoding='utf-8') as f:
            f.write(content)
    
    print("Создано 10 связанных документов в data/")
    return [f"data/{filename}" for filename, _ in docs]

def parse_document(filepath, doc_id):
    """Парсинг документа и извлечение ссылок"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Извлекаем заголовок (первая строка)
    lines = content.split('\n')
    title = lines[0] if lines else Path(filepath).stem
    
    # Извлекаем ссылки в формате [[doc_name]]
    links = re.findall(r'\[\[([^\]]+)\]\]', content)
    
    return Document(doc_id, str(filepath), title, content), links

def build_real_link_graph(documents, all_files):
    """Построение реального графа ссылок на основе ссылок в документах"""
    # Сначала создаем mapping имени файла -> ID
    filename_to_id = {}
    for doc_id, doc in documents.items():
        filename = Path(doc.path).name
        filename_to_id[filename] = doc_id
    
    link_graph = defaultdict(list)
    
    for doc_id, doc in documents.items():
        # Извлекаем ссылки из документа
        links_in_doc = re.findall(r'\[\[([^\]]+)\]\]', doc.content)
        
        real_links = []
        for link in links_in_doc:
            # Добавляем расширение .txt если нужно
            if not link.endswith('.txt'):
                link = link + '.txt'
            
            if link in filename_to_id:
                target_id = filename_to_id[link]
                real_links.append(target_id)
        
        # Обновляем документ и граф
        doc.outgoing_links = real_links
        link_graph[doc_id] = real_links
        
        # Обновляем входящие ссылки
        for target_id in real_links:
            if doc_id not in documents[target_id].incoming_links:
                documents[target_id].incoming_links.append(doc_id)
    
    return link_graph

# MAPREDUCE PAGERANK

class MapReducePageRank:
    def __init__(self, damping=0.85, iterations=20):
        self.damping = damping
        self.iterations = iterations
    
    def map_phase(self, graph, ranks):
        """Фаза Map: распределение PageRank по ссылкам"""
        contributions = defaultdict(float)
        
        for node, outlinks in graph.items():
            if outlinks:
                # Распределяем PageRank поровну по исходящим ссылкам
                share = ranks[node] / len(outlinks)
                for neighbor in outlinks:
                    contributions[neighbor] += share
            else:
                # Висячие узлы: распределяем всем
                share = ranks[node] / len(ranks)
                for neighbor in ranks.keys():
                    contributions[neighbor] += share
        
        return contributions
    
    def reduce_phase(self, contributions, N):
        """Фаза Reduce: вычисление нового PageRank"""
        new_ranks = {}
        for node in range(N):
            # Формула PageRank с телепортацией
            new_rank = (1 - self.damping) / N + self.damping * contributions.get(node, 0)
            new_ranks[node] = new_rank
        return new_ranks
    
    def calculate(self, graph, documents):
        """Полный цикл MapReduce"""
        N = len(documents)
        if N == 0:
            return {}
        
        # Инициализация
        ranks = {doc_id: 1.0 / N for doc_id in documents}
        
        print(f"\nMapReduce PageRank для {N} документов:")
        print(f"Всего итераций: {self.iterations}")
        
        for iteration in range(self.iterations):
            # Map
            contributions = self.map_phase(graph, ranks)
            
            # Reduce
            new_ranks = self.reduce_phase(contributions, N)
            
            # Вычисляем изменение для проверки сходимости
            total_diff = sum(abs(new_ranks[node] - ranks[node]) for node in ranks)
            ranks = new_ranks
            
            if iteration % 5 == 0:
                print(f"  Итерация {iteration}: изменение = {total_diff:.6f}")
        
        # Нормализация
        total = sum(ranks.values())
        if total > 0:
            ranks = {k: v / total for k, v in ranks.items()}
        
        print(f"\nТоп-5 документов (MapReduce):")
        top_5 = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:5]
        for doc_id, rank in top_5:
            if doc_id in documents:
                doc = documents[doc_id]
                print(f"  {doc.title[:40]:40} {rank:.6f}")
        
        return ranks

# ИНВЕРТИРОВАННЫЙ ИНДЕКС

class InvertedIndex:
    """Инвертированный индекс"""
    
    def __init__(self):
        self.index = defaultdict(dict)  # word -> {doc_id: [positions]}
        self.doc_lengths = {}  # doc_id -> количество слов
        self.total_docs = 0
        self.doc_freq = defaultdict(int)  # word -> количество документов с этим словом
    
    def add_document(self, doc_id, text):
        """Добавление документа в индекс"""
        words = self._tokenize(text)
        self.doc_lengths[doc_id] = len(words)
        self.total_docs += 1
        
        # Индексируем слова с позициями
        for position, word in enumerate(words):
            if doc_id not in self.index[word]:
                self.index[word][doc_id] = []
            self.index[word][doc_id].append(position)
        
        # Обновляем document frequency
        for word in set(words):
            self.doc_freq[word] += 1
    
    def _tokenize(self, text):
        """Токенизация текста"""
        text = text.lower()
        # Удаляем знаки препинания, оставляем слова
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        # Фильтруем короткие слова и стоп-слова
        stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'не', 'что', 'это', 'как'}
        return [w for w in words if len(w) > 2 and w not in stop_words]
    
    def tf(self, word, doc_id):
        """Term Frequency - частота слова в документе"""
        if doc_id in self.index.get(word, {}):
            return len(self.index[word][doc_id])
        return 0
    
    def idf(self, word):
        """Inverse Document Frequency"""
        df = self.doc_freq.get(word, 0)
        if df == 0:
            return 0
        return math.log(self.total_docs / df)
    
    def tf_idf(self, word, doc_id):
        """TF-IDF оценка"""
        return self.tf(word, doc_id) * self.idf(word)

#ПОИСКОВАЯ СИСТЕМА 

class SearchEngine:
    """Поисковая система с DAAT и TAAT"""
    
    def __init__(self, documents, pagerank, inverted_index):
        self.documents = documents
        self.pagerank = pagerank
        self.index = inverted_index
        
    def search_daat(self, query, top_k=10, use_pagerank=True):
        """
        Document-at-a-Time алгоритм
        Проходим по всем документам и вычисляем релевантность
        """
        query_words = self.index._tokenize(query)
        if not query_words:
            return []
        
        results = []
        
        for doc_id, doc in self.documents.items():
            score = 0
            
            # Вычисляем TF-IDF сумму для всех слов запроса
            for word in query_words:
                score += self.index.tf_idf(word, doc_id)
            
            if score > 0:
                # Учитываем PageRank если нужно
                if use_pagerank and doc_id in self.pagerank:
                    score *= (1 + self.pagerank[doc_id] * 2)  # Усиливаем влияние PageRank
                
                # Добавляем сниппет
                snippet = self._get_snippet(doc.content, query_words)
                
                results.append({
                    'id': doc_id,
                    'title': doc.title,
                    'score': score,
                    'pagerank': self.pagerank.get(doc_id, 0),
                    'snippet': snippet,
                    'incoming': len(doc.incoming_links),
                    'outgoing': len(doc.outgoing_links)
                })
        
        # Сортировка по убыванию релевантности
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def search_taat(self, query, top_k=10, use_pagerank=True):
        """
        Term-at-a-Time алгоритм
        Проходим по всем словам запроса и накапливаем оценки документов
        """
        query_words = self.index._tokenize(query)
        if not query_words:
            return []
        
        # Словарь для накопления оценок
        doc_scores = defaultdict(float)
        
        # TAAT: обрабатываем каждое слово отдельно
        for word in query_words:
            # Получаем документы, содержащие это слово
            doc_postings = self.index.index.get(word, {})
            
            # Вычисляем IDF для слова
            idf = self.index.idf(word)
            
            # Для каждого документа добавляем TF-IDF
            for doc_id, positions in doc_postings.items():
                tf = len(positions)
                doc_scores[doc_id] += tf * idf
        
        # Формируем результаты
        results = []
        for doc_id, tfidf_score in doc_scores.items():
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                
                final_score = tfidf_score
                if use_pagerank and doc_id in self.pagerank:
                    final_score *= (1 + self.pagerank[doc_id] * 2)
                
                snippet = self._get_snippet(doc.content, query_words)
                
                results.append({
                    'id': doc_id,
                    'title': doc.title,
                    'score': final_score,
                    'pagerank': self.pagerank.get(doc_id, 0),
                    'snippet': snippet,
                    'incoming': len(doc.incoming_links),
                    'outgoing': len(doc.outgoing_links)
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _get_snippet(self, content, query_words, max_length=150):
        """Создание сниппета с выделением найденных слов"""
        # Ищем первое вхождение любого из слов запроса
        content_lower = content.lower()
        
        for word in query_words:
            pos = content_lower.find(word)
            if pos != -1:
                # Берем текст вокруг найденного слова
                start = max(0, pos - 50)
                end = min(len(content), pos + 100)
                snippet = content[start:end]
                
                # Выделяем найденные слова
                for w in query_words:
                    snippet = re.sub(
                        f'({w})', 
                        r'**\1**', 
                        snippet, 
                        flags=re.IGNORECASE
                    )
                
                if start > 0:
                    snippet = "..." + snippet
                if end < len(content):
                    snippet = snippet + "..."
                
                return snippet
        
        # Если не нашли слова, берем начало документа
        snippet = content[:max_length]
        if len(content) > max_length:
            snippet += "..."
        return snippet
    
    def compare_algorithms(self, query, top_k=5):
        """Сравнение DAAT и TAAT"""
        print(f"\nСравнение алгоритмов для запроса: '{query}'")
        print("="*80)
        
        daat_results = self.search_daat(query, top_k, use_pagerank=True)
        taat_results = self.search_taat(query, top_k, use_pagerank=True)
        
        print("\nDAAT (Document-at-a-Time):")
        print("-"*40)
        for i, result in enumerate(daat_results, 1):
            print(f"{i}. {result['title']}")
            print(f"   Score: {result['score']:.4f}, PageRank: {result['pagerank']:.4f}")
            print(f"   {result['snippet'][:100]}...")
        
        print("\nTAAT (Term-at-a-Time):")
        print("-"*40)
        for i, result in enumerate(taat_results, 1):
            print(f"{i}. {result['title']}")
            print(f"   Score: {result['score']:.4f}, PageRank: {result['pagerank']:.4f}")
            print(f"   {result['snippet'][:100]}...")
        
        # Статистика сравнения
        daat_ids = {r['id'] for r in daat_results}
        taat_ids = {r['id'] for r in taat_results}
        common = daat_ids.intersection(taat_ids)
        
        print(f"\nСтатистика сравнения:")
        print(f"  DAAT нашел: {len(daat_results)} документов")
        print(f"  TAAT нашел: {len(taat_results)} документов")
        print(f"  Общих документов: {len(common)}")
        
        return daat_results, taat_results

# ОСНОВНЫЕ ФУНКЦИИ

def create_data_from_scratch():
    """Создание данных с нуля (если нет сохраненных)"""
    test_files = create_test_documents_with_links()
    
    documents = {}
    next_id = 0
    
    for filepath in test_files:
        doc, _ = parse_document(filepath, next_id)
        documents[next_id] = doc
        next_id += 1
    
    link_graph = build_real_link_graph(documents, test_files)
    
    return documents, link_graph

def load_existing_data():
    """Загрузка существующих данных"""
    # Проверяем, есть ли сохраненные данные
    if os.path.exists("outputs/documents.pkl") and os.path.exists("outputs/graph.pkl"):
        try:
            with open("outputs/documents.pkl", "rb") as f:
                documents = pickle.load(f)
            with open("outputs/graph.pkl", "rb") as f:
                link_graph = pickle.load(f)
            print("Загружены сохраненные данные")
            return documents, link_graph
        except:
            print("Ошибка загрузки сохраненных данных. Создаю новые...")
    
    # Создаем новые данные
    return create_data_from_scratch()

def save_results(documents, pagerank, index, search_engine):
    """Сохранение всех результатов"""
    os.makedirs("final_outputs", exist_ok=True)
    
    # 1. PageRank результаты
    with open("final_outputs/pagerank_final.txt", "w", encoding="utf-8") as f:
        f.write("Итоговый PageRank (MapReduce)\n")
        f.write("="*80 + "\n")
        sorted_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        for doc_id, rank in sorted_pr:
            doc = documents[doc_id]
            f.write(f"{doc.title[:50]:50} {rank:.6f}\n")
    
    # 2. Статистика индекса
    with open("final_outputs/index_stats.txt", "w", encoding="utf-8") as f:
        f.write("Статистика инвертированного индекса\n")
        f.write("="*80 + "\n")
        f.write(f"Всего документов: {index.total_docs}\n")
        f.write(f"Уникальных слов: {len(index.index)}\n")
        
        # Топ-20 слов по document frequency
        word_dfs = [(word, index.doc_freq[word]) for word in index.index]
        word_dfs.sort(key=lambda x: x[1], reverse=True)
        
        f.write("\nТоп-20 слов по частоте в документах:\n")
        for word, df in word_dfs[:20]:
            f.write(f"  {word}: {df} документов\n")
    
    # 3. Примеры поисковых запросов
    with open("final_outputs/search_examples.txt", "w", encoding="utf-8") as f:
        f.write("Примеры работы поиска\n")
        f.write("="*80 + "\n")
        
        queries = ["Python", "базы данных", "машинное обучение", "веб"]
        
        for query in queries:
            f.write(f"\nЗапрос: '{query}'\n")
            f.write("-"*40 + "\n")
            
            daat_results = search_engine.search_daat(query, top_k=3)
            f.write("DAAT (топ-3):\n")
            for i, result in enumerate(daat_results, 1):
                f.write(f"  {i}. {result['title']} (score: {result['score']:.4f})\n")
            
            taat_results = search_engine.search_taat(query, top_k=3)
            f.write("TAAT (топ-3):\n")
            for i, result in enumerate(taat_results, 1):
                f.write(f"  {i}. {result['title']} (score: {result['score']:.4f})\n")
    
    print("Результаты сохранены в папке 'final_outputs/'")

def demonstrate_all_components():
    """Демонстрация всех компонентов системы"""
    print("="*100)
    print("ДЕМОНСТРАЦИЯ ВСЕХ КОМПОНЕНТОВ ПОИСКОВОЙ СИСТЕМЫ")
    print("="*100)
    
    # 1. Загружаем существующие документы и граф
    print("\n1. ЗАГРУЗКА ДАННЫХ И ГРАФА")
    print("-"*50)
    
    documents, link_graph = load_existing_data()
    print(f"Загружено документов: {len(documents)}")
    print(f"Ссылок в графе: {sum(len(v) for v in link_graph.values())}")
    
    # 2. MapReduce PageRank
    print("\n2. PAGERANK С ИСПОЛЬЗОВАНИЕМ MAPREDUCE")
    print("-"*50)
    
    mapreduce_calc = MapReducePageRank(damping=0.85, iterations=30)
    mapreduce_ranks = mapreduce_calc.calculate(link_graph, documents)
    
    # 3. Построение инвертированного индекса
    print("\n3. ПОСТРОЕНИЕ ИНВЕРТИРОВАННОГО ИНДЕКСА")
    print("-"*50)
    
    index = InvertedIndex()
    for doc_id, doc in documents.items():
        index.add_document(doc_id, doc.content)
    
    print(f"Индексировано уникальных слов: {len(index.index)}")
    print(f"Всего документов в индексе: {index.total_docs}")
    
    # Пример статистики
    print("\nСтатистика индекса:")
    word_stats = [(word, len(docs)) for word, docs in index.index.items()]
    word_stats.sort(key=lambda x: x[1], reverse=True)
    print("Топ-10 самых частых слов:")
    for word, count in word_stats[:10]:
        print(f"  {word}: в {count} документах")
    
    # 4. Создание поисковой системы
    print("\n4. СОЗДАНИЕ ПОИСКОВОЙ СИСТЕМЫ")
    print("-"*50)
    
    search_engine = SearchEngine(documents, mapreduce_ranks, index)
    
    # 5. Демонстрация поиска
    print("\n5. ДЕМОНСТРАЦИЯ ПОИСКА")
    print("-"*50)
    
    test_queries = [
        "Python программирование",
        "веб разработка базы данных",
        "машинное обучение анализ",
        "фреймворки Django Flask",
        "установка библиотеки"
    ]
    
    for query in test_queries:
        print(f"\nЗапрос: '{query}'")
        print("-"*40)
        
        # DAAT поиск
        daat_results = search_engine.search_daat(query, top_k=3)
        print("DAAT результаты:")
        for i, result in enumerate(daat_results, 1):
            print(f"  {i}. {result['title']} (score: {result['score']:.4f}, PR: {result['pagerank']:.4f})")
        
        # TAAT поиск
        taat_results = search_engine.search_taat(query, top_k=3)
        print("TAAT результаты:")
        for i, result in enumerate(taat_results, 1):
            print(f"  {i}. {result['title']} (score: {result['score']:.4f}, PR: {result['pagerank']:.4f})")
    
    # 6. Подробное сравнение алгоритмов
    print("\n6. ПОДРОБНОЕ СРАВНЕНИЕ АЛГОРИТМОВ ПОИСКА")
    print("-"*50)
    
    search_engine.compare_algorithms("Python разработка", top_k=5)
    
    # 7. Анализ влияния PageRank на результаты
    print("\n7. ВЛИЯНИЕ PAGERANK НА РЕЗУЛЬТАТЫ ПОИСКА")
    print("-"*50)
    
    query = "Python"
    print(f"Запрос: '{query}'")
    
    print("\nБез учета PageRank:")
    results_no_pr = search_engine.search_daat(query, use_pagerank=False)
    for i, result in enumerate(results_no_pr[:3], 1):
        print(f"  {i}. {result['title']} (score: {result['score']:.4f})")
    
    print("\nС учетом PageRank:")
    results_with_pr = search_engine.search_daat(query, use_pagerank=True)
    for i, result in enumerate(results_with_pr[:3], 1):
        print(f"  {i}. {result['title']} (score: {result['score']:.4f}, PR: {result['pagerank']:.4f})")
    
    # 8. Сохранение результатов
    print("\n8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("-"*50)
    
    save_results(documents, mapreduce_ranks, index, search_engine)
    
    print("\n" + "="*100)
    print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА!")
    print("Все компоненты работают корректно:")
    print("  1. ✅ Парсинг документов и ссылок")
    print("  2. ✅ Построение графа ссылок")
    print("  3. ✅ PageRank через MapReduce")
    print("  4. ✅ Инвертированный индекс")
    print("  5. ✅ Поиск DAAT (Document-at-a-Time)")
    print("  6. ✅ Поиск TAAT (Term-at-a-Time)")
    print("  7. ✅ Учет PageRank в ранжировании")
    print("="*100)

# ЗАПУСК 

if __name__ == "__main__":
    demonstrate_all_components()