import os
import random
from pathlib import Path

def create_test_documents(num_docs=20, output_dir="data/"):
    """Создание тестовых документов со ссылками"""
    os.makedirs(output_dir, exist_ok=True)
    
    topics = [
        "программирование", "алгоритмы", "структуры данных",
        "машинное обучение", "веб разработка", "базы данных",
        "сети", "безопасность", "мобильные приложения"
    ]
    
    doc_names = []
    
    for i in range(num_docs):
        # Создаем документ
        topic = random.choice(topics)
        filename = f"doc_{i:03d}_{topic.replace(' ', '_')}.txt"
        doc_names.append(filename)
        
        # Генерируем содержание
        lines = [
            f"Документ {i}: {topic}",
            "=" * 40,
            "",
            f"Это документ о {topic}.",
            f"{topic.capitalize()} - важная тема в компьютерных науках.",
            "",
            "Основные понятия:",
            f"- Основы {topic}",
            f"- Применение {topic}",
            f"- Будущее {topic}",
            ""
        ]
        
        # Добавляем ссылки на другие документы
        if len(doc_names) > 1:
            lines.append("Связанные документы:")
            # Выбираем случайные документы для ссылок
            num_links = random.randint(1, min(3, len(doc_names)-1))
            linked_docs = random.sample(doc_names[:-1], num_links)
            
            for linked_doc in linked_docs:
                lines.append(f"- [[{linked_doc}]]")
        
        # Сохраняем файл
        filepath = Path(output_dir) / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    print(f"Создано {num_docs} тестовых документов в {output_dir}")

if __name__ == "__main__":
    create_test_documents(15)