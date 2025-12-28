"""
Индексатор документов и построитель графа ссылок
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from src.models import Document, InvertedIndex
from src.parser import ParserFactory

class Indexer:
    """Класс для индексации документов"""
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)
        self.documents: Dict[int, Document] = {}
        self.inverted_index = InvertedIndex()
        self.link_graph: Dict[int, List[int]] = defaultdict(list)
        self.doc_id_by_path: Dict[str, int] = {}
        self.next_id = 0
        
    def build_index(self, recursive: bool = True):
        """Построение индекса из всех документов в директории"""
        print("Начинаем индексацию документов...")
        
        # Сбор всех файлов
        file_patterns = ['.txt', '.html', '.htm', '.md', '.markdown']
        all_files = []
        
        for pattern in file_patterns:
            if recursive:
                files = list(self.data_dir.rglob(f"*{pattern}"))
            else:
                files = list(self.data_dir.glob(f"*{pattern}"))
            all_files.extend(files)
        
        print(f"Найдено файлов: {len(all_files)}")
        
        # Первый проход: парсинг и создание документов
        parsed_docs = {}
        for filepath in all_files:
            if self._should_skip_file(filepath):
                continue
                
            doc_id = self._parse_and_create_document(filepath)
            if doc_id is not None:
                parsed_docs[doc_id] = filepath
        
        # Второй проход: построение графа ссылок
        print("Построение графа ссылок...")
        self._build_link_graph(parsed_docs)
        
        print(f"Индексация завершена. Обработано документов: {len(self.documents)}")
        
    def _should_skip_file(self, filepath: Path) -> bool:
        """Проверка, нужно ли пропускать файл"""
        # Пропускаем скрытые файлы и файлы без расширения
        if filepath.name.startswith('.') or filepath.suffix == '':
            return True
        
        # Пропускаем слишком большие файлы (> 10MB)
        try:
            if filepath.stat().st_size > 10 * 1024 * 1024:
                print(f"Пропускаем слишком большой файл: {filepath}")
                return True
        except:
            pass
        
        return False
    
    def _parse_and_create_document(self, filepath: Path) -> Optional[int]:
        """Парсинг файла и создание документа"""
        try:
            # Получаем парсер для типа файла
            parser = ParserFactory.get_parser(filepath.suffix)
            if not parser:
                print(f"Неподдерживаемый формат: {filepath.suffix}")
                return None
            
            # Парсим файл
            text, links = parser.parse_file(filepath)
            
            # Создаем документ
            doc_id = self.next_id
            self.next_id += 1
            
            doc = Document(
                id=doc_id,
                path=str(filepath.relative_to(Path.cwd())),
                title=self._extract_title(filepath, text),
                content=text,
                word_count=len(text.split())
            )
            
            # Сохраняем документ
            self.documents[doc_id] = doc
            self.doc_id_by_path[str(filepath)] = doc_id
            
            # Индексируем текст
            tokens = parser.tokenize(text)
            self.inverted_index.add_document(doc_id, tokens)
            
            # Сохраняем сырые ссылки для последующей обработки
            doc.outgoing_links = links  # Временно храним как строки
            
            print(f"  Индексирован: {filepath.name} ({len(tokens)} слов)")
            return doc_id
            
        except Exception as e:
            print(f"Ошибка при индексации {filepath}: {e}")
            return None
    
    def _extract_title(self, filepath: Path, text: str) -> str:
        """Извлечение заголовка документа"""
        # Используем имя файла без расширения
        title = filepath.stem
        
        # Пытаемся найти заголовок в тексте
        lines = text.split('\n')
        for line in lines[:3]:  # Проверяем первые 3 строки
            line = line.strip()
            if len(line) > 10 and len(line) < 100:
                # Проверяем, похоже ли это на заголовок
                if not line.startswith(('#', '=', '-', '*')) and line.endswith(('.', '!', '?')):
                    title = line[:80]
                    break
        
        return title.replace('_', ' ').replace('-', ' ').title()
    
    def _build_link_graph(self, parsed_docs: Dict[int, Path]):
        """Построение графа ссылок между документами"""
        for doc_id, filepath in parsed_docs.items():
            doc = self.documents[doc_id]
            normalized_links = []
            
            for link in doc.outgoing_links:
                # Ищем документ по пути
                target_id = self._find_document_by_link(link, filepath)
                if target_id is not None and target_id != doc_id:
                    normalized_links.append(target_id)
            
            # Обновляем документ
            doc.outgoing_links = normalized_links
            
            # Добавляем в граф
            self.link_graph[doc_id] = normalized_links
            
            # Обновляем входящие ссылки в целевых документах
            for target_id in normalized_links:
                if target_id in self.documents:
                    if doc_id not in self.documents[target_id].incoming_links:
                        self.documents[target_id].incoming_links.append(doc_id)
    
    def _find_document_by_link(self, link: str, source_path: Path) -> Optional[int]:
        """Поиск ID документа по ссылке"""
        # Прямое совпадение по полному пути
        if link in self.doc_id_by_path:
            return self.doc_id_by_path[link]
        
        # Пробуем разные варианты
        link_variants = [
            link,
            link + '.txt',
            link + '.html',
            link + '.md',
            str(Path(link).with_suffix('.txt')),
            str(Path(link).with_suffix('.html')),
            str(Path(link).with_suffix('.md')),
        ]
        
        for variant in link_variants:
            if variant in self.doc_id_by_path:
                return self.doc_id_by_path[variant]
        
        # Попытка найти по имени файла
        link_filename = Path(link).name
        for path, doc_id in self.doc_id_by_path.items():
            if Path(path).name == link_filename:
                return doc_id
        
        return None
    
    def get_document_count(self) -> int:
        """Получить количество документов"""
        return len(self.documents)
    
    def get_total_words(self) -> int:
        """Получить общее количество индексированных слов"""
        total = 0
        for doc in self.documents.values():
            total += doc.word_count
        return total
    
    def save_index(self, output_dir: str = "outputs/"):
        """Сохранение индекса и данных на диск"""
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохраняем документы
        docs_data = {str(doc_id): doc.to_dict() for doc_id, doc in self.documents.items()}
        with open(f"{output_dir}/documents.json", 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)
        
        # Сохраняем граф
        graph_data = {str(k): v for k, v in self.link_graph.items()}
        with open(f"{output_dir}/graph.json", 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        # Сохраняем инвертированный индекс
        self.inverted_index.save(f"{output_dir}/index.json")
        
        print(f"Данные сохранены в директории {output_dir}")