import os
import re
import time
import shelve
import numpy as np
import ollama
from bs4 import BeautifulSoup
from libzim.reader import Archive
from libzim.search import Query, Searcher

from dotenv import load_dotenv
load_dotenv()

ZIM_FOLDER = os.getenv("ZIM_FOLDER_CODE","")
MODEL_NAME = os.getenv("MODEL_NAME_CODE","")
EMBED_MODEL = os.getenv("EMBED_MODEL_CODE","")

if not ZIM_FOLDER or not MODEL_NAME or not EMBED_MODEL:
    raise ValueError(f"All env varibles can't be empty. ZIM_FOLDER: {ZIM_FOLDER}, MODEL_NAME: {MODEL_NAME}, EMBED_MODEL: {EMBED_MODEL}")

def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return float(np.dot(v1, v2) / (norm1 * norm2)) if norm1 and norm2 else 0.0

def get_embedding_dynamic(text, cache_path):
    """Безопасно извлекает вектор из указанного файла кэша технологии."""
    text_key = text.strip()
    if not text_key: return None
    try:
        with shelve.open(cache_path) as cache:
            if text_key in cache:
                return cache[text_key]
            response = ollama.embeddings(model=EMBED_MODEL, prompt=text_key)
            embedding = response["embedding"]
            cache[text_key] = embedding
            return embedding
    except Exception:
        try:
            response = ollama.embeddings(model=EMBED_MODEL, prompt=text_key)
            return response["embedding"]
        except Exception:
            return None

def get_semantic_context(user_question):
    """Глобальный RAG: локальный перевод + сквозной поиск по ВСЕМ сепаратным индексам одновременно."""
    try:
        zim_files = [f for f in os.listdir(ZIM_FOLDER) if f.endswith('.zim')]
    except Exception:
        return ""
        
    # 1. ПОЛНОСТЬЮ ЛОКАЛЬНЫЙ ОФЛАЙН-ПЕРЕВОД ВОПРОСА СИЛАМИ OLLAMA
    try:
        print(" ⏳ Локальный перевод вопроса через Ollama (офлайн)...", flush=True)
        translate_prompt = (
            f"Translate the following text from Russian to English. "
            f"Return ONLY the translation text, without any quotes, explanations, headers or extra words. "
            f"Do not write anything except the translated English text:\n\n{user_question}"
        )
        translation_response = ollama.chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': translate_prompt}],
            options={"temperature": 0.0}
        )
        translated_query = translation_response['message']['content'].strip()
        translated_query = translated_query.replace('"', '').replace("'", "")
        print(f" Локальный перевод вопроса на английский: 🇬🇧 '{translated_query}'", flush=True)
    except Exception as e:
        print(f"⚠️ Ошибка локального перевода: {e}. Использую оригинал.", flush=True)
        translated_query = user_question

    # Вычисляем вектор вопроса (сохраняем в кэш вопросов cache_main_query)
    indexes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indexes")
    main_query_cache = os.path.join(indexes_dir, "cache_main_query")
    question_vector = get_embedding_dynamic(translated_query, main_query_cache)
    if not question_vector:
        return ""

    clean_text = re.sub(r'[^\w\s]', ' ', translated_query.lower())
    search_words = [w for w in clean_text.split() if len(w) > 2]
    if not search_words:
        return ""
    
    candidates_found = 0
    candidates_pool = []

    # 2. СКВОЗНОЙ СКАНЕР: Проходим по всем ZIM-файлам и собираем чанки в общий пул
    for zim_file in zim_files:
        if "wikipedia" in zim_file.lower():
            continue  # Википедию не трогаем, у нее свой скрипт wiki.py
            
        zim_path = os.path.join(ZIM_FOLDER, zim_file)
        
        # Динамически вычисляем путь к сепарной базе кэша для этого файла
        clean_name = zim_file.replace("-", "_").replace(".", "_")
        current_cache_path = os.path.join(indexes_dir, f"cache_{clean_name}")
        
        try:
            archive = Archive(zim_path) # type: ignore
            searcher = Searcher(archive)
            file_candidates = 0
            
            for main_word in search_words:
                # Из каждого файла берем до 30 статей, чтобы не перегружать память, но собрать максимум
                if file_candidates >= 30:
                    break
                    
                query = Query().set_query(main_word)
                search = searcher.search(query)
                paths = list(search.getResults(0, 20))
                if not paths:
                    continue
                    
                for path in paths:
                    if file_candidates >= 30:
                        break

                    try:
                        entry = archive.get_entry_by_path(path)
                        if any(entry.path.lower().endswith(ext) for ext in ['.css', '.js', '.png', '.jpg', '.svg', '.woff']):
                            continue
                            
                        raw_bytes = bytes(entry.get_item().content)
                        html_text = raw_bytes.decode('utf-8', errors='ignore')
                        
                        soup = BeautifulSoup(html_text, 'html.parser')
                        for element in soup(["script", "style", "nav", "footer", "header"]):
                            element.extract()
                            
                        body_text = " ".join([line.strip() for line in soup.get_text().splitlines() if line.strip()])
                        if len(body_text) < 150: 
                            continue
                            
                        chunk = body_text[:1500]
                        
                        # Тянем вектор из персонального кэша текущей открытой технологии!
                        chunk_vector = get_embedding_dynamic(chunk, current_cache_path)
                        
                        if chunk_vector:
                            file_candidates += 1
                            candidates_found += 1
                            score = cosine_similarity(question_vector, chunk_vector)
                            
                            candidates_pool.append({
                                "score": score,
                                "text": chunk,
                                "source": zim_file,
                                "title": entry.title
                            })
                    except Exception:
                        continue
        except Exception:
            continue

    print(f"📊 Глобальный поиск завершен. Собрано и оценено: {candidates_found} чанков со всех баз.", flush=True)

    # 3. ГЛОБАЛЬНАЯ СОРТИРОВКА: NumPy выбирает 3 лучших куска текста СРЕДИ ВСЕХ ТЕХНОЛОГИЙ
    combined_context = []
    if candidates_pool:
        candidates_pool.sort(key=lambda x: x["score"], reverse=True)
        
        print("\n🏆 === ГЛОБАЛЬНЫЕ ВЕКТОРНЫЕ СОВПАДЕНИЯ (NumPy) ===", flush=True)
        top_candidates = candidates_pool[:3] # Берем топ-3 лучших совпадения в мире
        for idx, cand in enumerate(top_candidates):
            print(f"[{idx+1}] Статья: '{cand['title']}' | База: {cand['source']} | Сходство: {cand['score']:.4f}", flush=True)
            combined_context.append(f"[Источник: {cand['source']} | Тема: {cand['title']}]\n{cand['text']}")
        print("===============================================\n", flush=True)

    return "\n\n".join(combined_context)[:6000]

def start_pro_rag_chat(): 
    print("=" * 60, flush=True) 
    print(" 🚀 ГЛОБАЛЬНЫЙ ТЕХНИЧЕСКИЙ ОФЛАЙН-ЧАТ (ВСЕ ИНДЕКСЫ) ЗАПУЩЕН") 
    print(f"• Папка назначения для индексов: {os.path.join(ZIM_FOLDER, 'indexes')}") 
    print(f"• Модель генерации: {MODEL_NAME}") 
    print("=" * 60, flush=True)
    
    system_instruction = (
        "Ты — сверхопытный локальный инженер-программист, работающий строго по предоставленной технической документации. "
        "Отвечай на вопрос пользователя развернуто, понятно и подробно, используя предоставленные факты. "
        "Обязательно выведи точную консольную команду или пример кода, если они есть в тексте. "
        "Отвечай на русском языке."
    )

    while True:
        start_time = time.time()
        try:
            print("\n" + "_"*40, flush=True)
            question = input("\n Ваш инженерный вопрос: ").strip()
            if not question: continue
            if question.lower() == 'exit': break
             
            print(" Шаг 1: Сканирование всех сепаратных кэшей и оценка смыслов...", flush=True) 
            first_step_time = time.time()
            context = get_semantic_context(question)
             
            if not context.strip():
                print(" Совпадений по смыслу в локальных базах не найдено.", flush=True) 
                continue

            print(f"⏱ Время выполнения первого шага: {time.time()-first_step_time:.2f} сек.", flush=True)
                 
            print(f" Шаг 2: Направляю точечные данные в YandexGPT для финального ответа...", flush=True)
            user_prompt = f"ДОКУМЕНТАЦИЯ ДЛЯ АНАЛИЗА:\n{context}\n\nВОПРОС: {question}"
             
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
            print(f"\n Ошибка: {e}", flush=True) 
        print(f"⏱ Время обработки всего запроса: {time.time()-start_time:.2f} сек.", flush=True)

if __name__ == '__main__':
    start_pro_rag_chat()
