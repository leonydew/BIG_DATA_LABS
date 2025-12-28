"""
Модели данных для поисковой системы
"""
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any, DefaultDict
from collections import defaultdict
import json

@dataclass
class Document:
    """Класс документа"""
    id: int
    path: str
    title: str
    content: str
    outgoing_links: List[int] = field(default_factory=list)
    incoming_links: List[int] = field(default_factory=list)
    word_count: int = 0
    pagerank: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'id': self.id,
            'path': self.path,
            'title': self.title[:100],
            'content_length': len(self.content),
            'outgoing_links': len(self.outgoing_links),
            'incoming_links': len(self.incoming_links),
            'word_count': self.word_count,
            'pagerank': self.pagerank
        }

class InvertedIndex:
    """Инвертированный индекс"""
    
    def __init__(self):
        self.index: DefaultDict[str, DefaultDict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.doc_freq: Dict[str, int] = defaultdict(int)  # Document frequency
        self.total_docs = 0
        
    def add_document(self, doc_id: int, tokens: List[str]):
        """Добавление документа в индекс"""
        self.total_docs += 1
        
        for position, token in enumerate(tokens):
            self.index[token][doc_id].append(position)
        
        # Обновляем document frequency
        unique_tokens = set(tokens)
        for token in unique_tokens:
            self.doc_freq[token] += 1
    
    def get_postings(self, token: str) -> Dict[int, List[int]]:
        """Получить постинги для токена"""
        return dict(self.index.get(token, {}))
    
    def get_document_frequency(self, token: str) -> int:
        """Получить частоту документа для токена"""
        return self.doc_freq.get(token, 0)
    
    def save(self, filename: str):
        """Сохранение индекса в файл"""
        # Конвертируем defaultdict в dict для сериализации
        index_dict = {}
        for token, postings in self.index.items():
            index_dict[token] = dict(postings)
            
        data = {
            'index': index_dict,
            'doc_freq': dict(self.doc_freq),
            'total_docs': self.total_docs
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filename: str):
        """Загрузка индекса из файла"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.index = defaultdict(lambda: defaultdict(list))
        for token, postings in data['index'].items():
            self.index[token].update(postings)
        
        self.doc_freq = defaultdict(int, data['doc_freq'])
        self.total_docs = data['total_docs']

class SearchResult:
    """Результат поиска"""
    
    def __init__(self, doc_id: int, score: float, document: Document, snippets: List[str] = None):
        self.doc_id = doc_id
        self.score = score
        self.document = document
        self.snippets = snippets or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'id': self.doc_id,
            'score': round(self.score, 4),
            'title': self.document.title,
            'path': self.document.path,
            'snippets': self.snippets[:2]  # Только первые 2 сниппета
        }