import os
import re
import time
import shelve
import numpy as np
import ollama
from libzim.reader import Archive
from libzim.search import Query, Searcher

from dotenv import load_dotenv
load_dotenv()

ZIM_FOLDER = os.getenv("ZIM_FOLDER_WIKI","")
WIKI_ZIM_FILE = os.getenv("ZIM_FILE_WIKI","")
WIKI_ZIM_PATH = os.getenv("ZIM_FOLDER_WIKI","")

CACHE_FILE_NAME = os.getenv("CACHE_FILE_WIKI","")

MODEL_NAME = os.getenv("MODEL_NAME_WIKI","")
EMBED_MODEL = os.getenv("EMBED_MODEL_WIKI","")

if not ZIM_FOLDER or not WIKI_ZIM_FILE or not EMBED_MODEL or not WIKI_ZIM_PATH or not CACHE_FILE_NAME or not MODEL_NAME:
    raise ValueError(f"All env varibles can't be empty. ZIM_FOLDER: {ZIM_FOLDER}, WIKI_ZIM_FILE: {WIKI_ZIM_FILE}, EMBED_MODEL: {EMBED_MODEL}, WIKI_ZIM_PATH: {WIKI_ZIM_PATH}, CACHE_FILE_NAME: {CACHE_FILE_NAME}, MODEL_NAME: {MODEL_NAME}")

WIKI_ZIM_PATH = os.path.join(ZIM_FOLDER, WIKI_ZIM_FILE)
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), CACHE_FILE_NAME)

# 1. Обновленная функция get_embedding (с выводом точной причины падения Ollama)
def get_embedding(text):
    """Генерирует вектор через Ollama с сохранением в локальный wiki-кэш."""
    text_key = text.strip()
    if not text_key:
        return None
    try:
        with shelve.open(CACHE_FILE) as cache:
            if text_key in cache:
                return cache[text_key]
            
            response = ollama.embeddings(model=EMBED_MODEL, prompt=text_key)
            embedding = response["embedding"]
            cache[text_key] = embedding
            return embedding
    except Exception as e:
        # Выводим реальную ошибку Ollama в консоль, чтобы сразу увидеть проблему!
        print(f" \n   [Ошибка Ollama API: {e}] ", end="", flush=True)
        try:
            response = ollama.embeddings(model=EMBED_MODEL, prompt=text_key)
            return response["embedding"]
        except Exception:
            return None


def cosine_similarity(v1, v2):
    """Вычисляет косинусное сходство между двумя векторами."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

# 2. Обновленная функция get_semantic_context (с уменьшенным размером чанков под мультиязычную модель)
def get_semantic_context(user_question):
    """Чистый поиск по русской Википедии на русском языке + Метод скользящего окна."""
    if not os.path.exists(WIKI_ZIM_PATH):
        print(f"⚠️ Ошибка: Файл базы знаний не найден по адресу: {WIKI_ZIM_PATH}", flush=True)
        return ""
        
    question_vector = get_embedding(user_question)
    if not question_vector:
        print("⚠️ Не удалось сгенерировать вектор вопроса.", flush=True)
        return ""

    # Нормализация текста: замена ё -> е для индекса Kiwix
    normalized_text = user_question.lower().replace('ё', 'е')
    clean_text = re.sub(r'[^\w\s]', ' ', normalized_text)
    
    STOP_WORDS = {"кто", "что", "когда", "где", "как", "почему", "зачем", "был", "была", "было", "были", "он", "она", "они"}
    search_words = [w for w in clean_text.split() if len(w) > 2 and w not in STOP_WORDS]
    
    if not search_words:
        print("⚠️ После фильтрации не осталось ключевых слов для поиска.", flush=True)
        return ""

    search_words.sort(key=len, reverse=True)
    important_words = search_words[:3]
    
    print(f"🔍 Ключевые слова для сканирования индекса: {important_words}", flush=True)
    
    candidates_found = 0
    candidates_pool = []

    try:
        archive = Archive(WIKI_ZIM_PATH) # type: ignore
        searcher = Searcher(archive)
        
        for main_word in important_words:
            if candidates_found >= 50: # Немного расширим общий пул
                break
                
            print(f"  👉 Запрос к индексу по слову: '{main_word}'", flush=True)
            query = Query().set_query(main_word)
            search = searcher.search(query)
            
            paths = list(search.getResults(0, 15))
            print(f"   🔎 Index вернул {len(paths)} потенциальных путей для слова '{main_word}'", flush=True)
            
            if not paths:
                continue
                
            for path in paths:
                if candidates_found >= 50:
                    break

                print(f"   ℹ️ Читаю и обрабатываю статью по пути: {path}...", end="", flush=True)

                try:
                    entry = archive.get_entry_by_path(path)
                    
                    if any(entry.path.lower().endswith(ext) for ext in ['.css', '.js', '.png', '.jpg', '.svg', '.woff']):
                        print(" ⏩ Пропущено (медиа/скрипт)", flush=True)
                        continue
                        
                    raw_bytes = bytes(entry.get_item().content)
                    html_text = raw_bytes.decode('utf-8', errors='ignore')
                    
                    # Глубокая очистка от HTML-разметки и таблиц
                    html_clean = re.sub(r'<(script|style|table|tr|td|th|nav|footer|header)[^>]*>.*?</\1>', '', html_text, flags=re.DOTALL | re.IGNORECASE)
                    text_only = re.sub(r'<[^>]+>', ' ', html_clean)
                    clean_body = " ".join([line.strip() for line in text_only.splitlines() if line.strip()])
                    clean_body = re.sub(r'\s+', ' ', clean_body).strip()
                    
                    # Разбиваем весь чистый текст статьи на слова
                    all_words = clean_body.split()
                    
                    if len(all_words) < 15: 
                        print(f" ⏩ Пропущено (мало слов: {len(all_words)})", flush=True)
                        continue
                        
                    # === МЕХАНИЗМ СКОЛЬЗЯЩЕГО ОКНА ЧЕРЕЗ КАЖДЫЕ 25 СЛОВ ===
                    # Сканируем первые 4 окна (кусочка) статьи, чтобы добраться до исторических фактов
                    # Окна: 0-25, 25-50, 50-75, 75-100 слов
                    window_size = 25
                    max_windows = 12
                    
                    article_candidates = 0
                    
                    for i in range(0, min(len(all_words), window_size * max_windows), window_size):
                        chunk_words = all_words[i : i + window_size]
                        if len(chunk_words) < 10:
                            continue
                            
                        chunk = " ".join(chunk_words)
                        chunk_vector = get_embedding(chunk)
                        
                        if chunk_vector:
                            score = cosine_similarity(question_vector, chunk_vector)
                            article_candidates += 1
                            candidates_found += 1
                            
                            candidates_pool.append({
                                "score": score,
                                "text": chunk,
                                "source": WIKI_ZIM_FILE,
                                "title": f"{entry.title} (Часть {article_candidates})"
                            })
                            
                    if article_candidates > 0:
                        print(f" ✅ Готово! Разбили '{entry.title}' на {article_candidates} безопасных окон.", flush=True)
                    else:
                        print(f" ⚠️ Не удалось извлечь окна из '{entry.title}'", flush=True)
                        
                except Exception as e:
                    print(f" ❌ Ошибка: {e}", flush=True)
                    continue
    except Exception as e:
        print(f"💥 Критическая ошибка при чтении ZIM-индекса: {e}", flush=True)
        return ""

    print(f"📊 Всего успешно сохранено и оценено по смыслу: {candidates_found} чанков.", flush=True)

        # 3. УМНАЯ ГРУППИРОВКА: Собираем связный текст вокруг статьи-победителя
    combined_context = []
    if candidates_pool:
        # Сортируем, чтобы найти абсолютного лидера по смыслу
        candidates_pool.sort(key=lambda x: x["score"], reverse=True)
        best_candidate = candidates_pool[0]
        
        # Находим имя оригинальной статьи (убираем приписку "Часть X")
        best_title = best_candidate["title"].split(" (Часть")[0]
        print(f"\n🏆 Абсолютный лидер по смыслу: '{best_title}' (Сходство: {best_candidate['score']:.4f})", flush=True)
        
        # Собираем ВСЕ окна, которые принадлежат этой статье-победителю
        # (Они отсортируются по порядку их чтения i, а не по скору!)
        article_windows = [cand for cand in candidates_pool if cand["title"].startswith(best_title)]
        
        print("\n🎯 === ВЕКТОРНЫЕ СОВПАДЕНИЯ (Умная группировка) ===", flush=True)
        print(f" Склеиваем {len(article_windows)} соседних окон для статьи '{best_title}' в связный текст.", flush=True)
        print("=======================================\n", flush=True)
        
        # Склеиваем их в один плотный, идущий по порядку исторический текст
        full_article_text = " ".join([cand["text"] for cand in article_windows])
        combined_context.append(f"[Источник: {WIKI_ZIM_FILE} | Тема: {best_title}]\n{full_article_text}")

    return "\n\n".join(combined_context)[:6000]


def start_pro_rag_chat(): 
    print("=" * 60, flush=True) 
    print(" 📚 ЧЕСТНАЯ ВЕКТОРНАЯ RAG-СИСТЕМА (ВИКИПЕДИЯ) ЗАПУЩЕНА", flush=True) 
    print(f"• Файл базы знаний: {WIKI_ZIM_PATH}", flush=True) 
    print(f"• Генератор ответов: {MODEL_NAME}", flush=True) 
    print(f"• Модель эмбеддингов: {EMBED_MODEL}", flush=True)
    print("=" * 60, flush=True)

    system_instruction = (
        "Ты — эрудированный локальный ИИ-ассистент, работающий строго по предоставленным статьям Википедии. "
        "Отвечай на вопрос пользователя развернуто, академично, понятно и подробно, используя предоставленные факты. "
        "Отвечай на русском языке."
    )

    while True:
        start_time = time.time()
        try:
            print("\n" + "_"*40, flush=True)
            question = input("\n Ваш вопрос Википедии: ").strip()
            if not question: continue
            if question.lower() == 'exit': break
             
            print("  Шаг 1: Сканирование индекса и оценка смыслов...", flush=True) 
            context = get_semantic_context(question)
             
            if not context.strip():
                print("  Совпадений по смыслу в локальных базах не найдено.", flush=True) 
                continue
                 
            print(f" 🧠 Шаг 2: Направляю точечные данные в YandexGPT для ответа...", flush=True)
             
            user_prompt = f"ВЫДЕРЖКИ ИЗ ВИКИПЕДИИ ДЛЯ АНАЛИЗА:\n{context}\n\nВОПРОС: {question}"
             
            response_stream = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {'role': 'system', 'content': system_instruction},
                    {'role': 'user', 'content': user_prompt}
                ],
                stream=True 
            )
            print("\n ✨ ОТВЕТ ИИ: ", end="", flush=True) 
            for chunk in response_stream:
                print(chunk['message']['content'], end="", flush=True)
            print() 
             
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n 💥 Ошибка: {e}", flush=True) 
        print(f"⏱ Время выполнения всего запроса: {time.time() - start_time:.2f} сек.", flush=True)

if __name__ == "__main__": 
    start_pro_rag_chat()
