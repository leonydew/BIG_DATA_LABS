"""
Реализация полнотекстового поиска: DAAT и TAAT
"""
import re
import math
from collections import defaultdict

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