"""
Парсер документов различных форматов
"""
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import html
from urllib.parse import urljoin
from collections import defaultdict

class BaseParser:
    """Базовый класс парсера"""
    
    def __init__(self):
        self.stop_words = self._load_stop_words()
        self.link_pattern = re.compile(r'\[\[([^\]]+)\]\]')
    
    def _load_stop_words(self) -> Set[str]:
        """Загрузка стоп-слов"""
        return set([
            'и', 'в', 'на', 'с', 'по', 'для', 'не', 'что', 'это', 'как',
            'а', 'то', 'но', 'из', 'от', 'до', 'же', 'бы', 'вы', 'за',
            'к', 'о', 'у', 'со', 'ли', 'ну', 'вот', 'ведь', 'там', 'тут'
        ])
    
    def parse_file(self, filepath: Path) -> Tuple[str, List[str]]:
        """Парсинг файла - должен быть переопределен в подклассах"""
        raise NotImplementedError
    
    def tokenize(self, text: str) -> List[str]:
        """Токенизация текста"""
        # Очистка и нормализация
        text = text.lower()
        text = re.sub(r'[^\w\s\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Разбиение на слова
        words = text.split()
        
        # Фильтрация
        filtered = []
        for word in words:
            word = word.strip('-_')
            if (len(word) > 2 and 
                word not in self.stop_words and
                not word.isdigit()):
                filtered.append(word)
        
        return filtered
    
    def extract_links(self, content: str, base_path: str = None) -> List[str]:
        """Извлечение ссылок из текста"""
        links = []
        
        # Извлечение ссылок в формате [[doc_name]]
        matches = self.link_pattern.findall(content)
        links.extend(matches)
        
        # Извлечение HTML ссылок
        html_links = re.findall(r'<a[^>]*href=["\']([^"\']+)["\'][^>]*>', content)
        links.extend(html_links)
        
        # Извлечение Markdown ссылок
        md_links = re.findall(r'\[([^\]]*)\]\(([^)]+)\)', content)
        for _, link in md_links:
            links.append(link)
        
        # Нормализация ссылок
        normalized_links = []
        for link in links:
            if base_path and not link.startswith(('http://', 'https://', '/')):
                # Преобразование относительных путей
                normalized = self._normalize_link(link, base_path)
                if normalized:
                    normalized_links.append(normalized)
            else:
                normalized_links.append(link)
        
        return normalized_links
    
    def _normalize_link(self, link: str, base_path: str) -> Optional[str]:
        """Нормализация относительной ссылки"""
        try:
            # Удаление якорей и параметров
            link = link.split('#')[0].split('?')[0]
            
            # Если ссылка относительная
            if not link.startswith('/'):
                # Добавляем расширение .txt, если его нет
                if not Path(link).suffix:
                    link = link + '.txt'
                
                # Построение полного пути относительно базового
                base_dir = Path(base_path).parent
                full_path = (base_dir / link).resolve()
                
                # Проверка существования файла
                if full_path.exists():
                    return str(full_path.relative_to(Path.cwd()))
        
        except Exception as e:
            print(f"Ошибка нормализации ссылки {link}: {e}")
        
        return None

class TextParser(BaseParser):
    """Парсер текстовых файлов"""
    
    def parse_file(self, filepath: Path) -> Tuple[str, List[str]]:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Извлекаем текст (удаляем ссылки в формате [[...]])
        text = re.sub(r'\[\[[^\]]+\]\]', '', content)
        
        # Извлекаем ссылки
        links = self.extract_links(content, str(filepath))
        
        return text, links

class HTMLParser(BaseParser):
    """Парсер HTML файлов"""
    
    def parse_file(self, filepath: Path) -> Tuple[str, List[str]]:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Извлечение текста из HTML
        text = self._extract_text_from_html(content)
        
        # Извлечение ссылок
        links = self.extract_links(content, str(filepath))
        
        return text, links
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """Извлечение чистого текста из HTML"""
        # Удаляем теги скриптов и стилей
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL)
        
        # Заменяем теги на пробелы
        html_content = re.sub(r'<[^>]+>', ' ', html_content)
        
        # Декодируем HTML-сущности
        text = html.unescape(html_content)
        
        # Удаляем лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

class MarkdownParser(BaseParser):
    """Парсер Markdown файлов"""
    
    def parse_file(self, filepath: Path) -> Tuple[str, List[str]]:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Извлечение текста (упрощенно - удаляем Markdown разметку)
        text = self._extract_text_from_markdown(content)
        
        # Извлечение ссылок
        links = self.extract_links(content, str(filepath))
        
        return text, links
    
    def _extract_text_from_markdown(self, markdown_content: str) -> str:
        """Извлечение текста из Markdown"""
        # Удаляем заголовки
        text = re.sub(r'#+\s*', '', markdown_content)
        
        # Удаляем Markdown разметку
        text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)  # Жирный и курсив
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Код
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)  # Изображения
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Ссылки без URL
        
        # Удаляем блочные элементы
        text = re.sub(r'^>+\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^-\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)
        
        return text.strip()

class ParserFactory:
    """Фабрика парсеров"""
    
    PARSERS = {
        '.txt': TextParser,
        '.html': HTMLParser,
        '.htm': HTMLParser,
        '.md': MarkdownParser,
        '.markdown': MarkdownParser
    }
    
    @staticmethod
    def get_parser(file_extension: str) -> Optional[BaseParser]:
        """Получить парсер по расширению файла"""
        parser_class = ParserFactory.PARSERS.get(file_extension.lower())
        if parser_class:
            return parser_class()
        return None