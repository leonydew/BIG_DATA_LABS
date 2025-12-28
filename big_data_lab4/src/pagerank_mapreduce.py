"""
PageRank с использованием парадигмы MapReduce
"""
from collections import defaultdict

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
    
    def simulate_distributed(self, graph, documents, num_workers=3):
        """Имитация распределенного MapReduce"""
        print(f"\nИмитация распределенного MapReduce ({num_workers} workers)...")
        
        N = len(documents)
        ranks = {doc_id: 1.0 / N for doc_id in documents}
        
        # Разделяем граф между workers
        nodes = list(graph.keys())
        chunk_size = len(nodes) // num_workers + 1
        
        for iteration in range(self.iterations):
            # Распределенная фаза Map
            all_contributions = defaultdict(float)
            
            for worker in range(num_workers):
                start = worker * chunk_size
                end = min((worker + 1) * chunk_size, len(nodes))
                worker_nodes = nodes[start:end]
                
                # Каждый worker обрабатывает свою часть
                worker_contributions = defaultdict(float)
                for node in worker_nodes:
                    if node in graph:
                        outlinks = graph[node]
                        if outlinks:
                            share = ranks[node] / len(outlinks)
                            for neighbor in outlinks:
                                worker_contributions[neighbor] += share
                        else:
                            share = ranks[node] / N
                            for neighbor in documents:
                                worker_contributions[neighbor] += share
                
                # Объединяем результаты от всех workers
                for node, contribution in worker_contributions.items():
                    all_contributions[node] += contribution
            
            # Фаза Reduce (централизованная)
            new_ranks = {}
            for node in documents:
                new_ranks[node] = (1 - self.damping) / N + self.damping * all_contributions.get(node, 0)
            
            ranks = new_ranks
        
        print("Распределенный MapReduce завершен")
        return ranks