"""
Реализация алгоритма PageRank с использованием графовой библиотеки NetworkX
(Аналог Pregel для небольших графов)
"""
import networkx as nx
from collections import defaultdict


class GraphPageRank:
    """PageRank с использованием NetworkX (Pregel-подобный подход)"""
    
    def __init__(self, damping_factor: float = 0.85, max_iterations: int = 100):
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
    
    def calculate(self, graph, documents):
        """Вычисление PageRank с использованием NetworkX"""
        # Создание ориентированного графа
        G = nx.DiGraph()
        
        # Добавление узлов и ребер
        print("Построение графа для NetworkX...")
        
        # Добавляем все узлы (даже изолированные)
        for doc_id in documents.keys():
            G.add_node(doc_id)
        
        # Добавляем ребра
        edge_count = 0
        for source, targets in graph.items():
            for target in targets:
                if target in documents:  # Проверяем, что целевой документ существует
                    G.add_edge(source, target)
                    edge_count += 1
        
        print(f"Граф построен: {G.number_of_nodes()} узлов, {edge_count} ребер")
        
        # Вычисление PageRank
        print("Вычисление PageRank с использованием NetworkX...")
        
        # Вариант 1: Стандартный PageRank
        pagerank = nx.pagerank(
            G, 
            alpha=self.damping_factor,
            max_iter=self.max_iterations,
            tol=1e-6
        )
        
        # Присваиваем ранги документам
        for doc_id, rank in pagerank.items():
            if doc_id in documents:
                documents[doc_id].pagerank = rank
        
        print(f"PageRank вычислен. Топ-5 документов (NetworkX):")
        top_5 = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
        for doc_id, rank in top_5:
            if doc_id in documents:
                doc = documents[doc_id]
                print(f"  {doc.title[:40]:40} {rank:.6f}")
        
        return pagerank
    
    def calculate_pregel_style(self, graph, documents):
        """
        PageRank в стиле Pregel (итеративный подход с передачей сообщений)
        Имитация модели Pregel на небольших графах
        """
        N = len(documents)
        if N == 0:
            return {}
        
        # Инициализация
        ranks = {doc_id: 1.0 / N for doc_id in documents.keys()}
        dangling_nodes = [node for node, outlinks in graph.items() if not outlinks]
        
        print(f"Вычисление PageRank (Pregel-стиль) для {N} документов...")
        
        for iteration in range(self.max_iterations):
            # Фаза 1: Каждый узел отправляет свой PageRank соседям
            messages = defaultdict(float)
            
            for node, outlinks in graph.items():
                if outlinks:
                    # Отправка доли PageRank каждому соседу
                    share = ranks[node] / len(outlinks)
                    for neighbor in outlinks:
                        messages[neighbor] += share
                else:
                    # Висячие узлы: распределяем равномерно
                    share = ranks[node] / N
                    for neighbor in documents.keys():
                        messages[neighbor] += share
            
            # Фаза 2: Каждый узел вычисляет новый PageRank
            new_ranks = {}
            total_diff = 0.0
            
            for node in documents.keys():
                received = messages.get(node, 0)
                new_rank = (1 - self.damping_factor) / N + self.damping_factor * received
                new_ranks[node] = new_rank
                total_diff += abs(new_rank - ranks[node])
            
            # Обновление
            ranks = new_ranks
            
            # Проверка сходимости
            avg_diff = total_diff / N
            if iteration % 10 == 0:
                print(f"  Итерация {iteration}: среднее изменение = {avg_diff:.6f}")
            
            if avg_diff < 1e-6:
                print(f"  Сходимость достигнута на итерации {iteration}")
                break
        
        # Нормализация
        rank_sum = sum(ranks.values())
        if rank_sum > 0:
            ranks = {k: v / rank_sum for k, v in ranks.items()}
        
        return ranks