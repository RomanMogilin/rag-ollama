import os
import shelve
import time
import string
from bs4 import BeautifulSoup
from libzim.reader import Archive
from libzim.search import Query, Searcher 
import ollama

from dotenv import load_dotenv
load_dotenv()

TARGET_ZIMS_FILES = os.getenv("TARGET_ZIMS_FILES","")
TARGET_ZIMS_FOLDER = os.getenv("TARGET_ZIMS_FOLDER","")
EMBED_MODEL = os.getenv("EMBED_MODEL_INDEX","")
OUT_INDEX_FOLDER = os.getenv("OUT_INDEX_FOLDER","")

if not TARGET_ZIMS_FILES or not TARGET_ZIMS_FOLDER or not EMBED_MODEL or not OUT_INDEX_FOLDER:
    raise ValueError(f"All env varibles can't be empty. TARGET_ZIMS_FOLDER: {TARGET_ZIMS_FOLDER}, TARGET_ZIMS_FILES: {TARGET_ZIMS_FILES}, EMBED_MODEL: {EMBED_MODEL}, OUT_INDEX_FOLDER: {OUT_INDEX_FOLDER}")

TARGET_ZIMS = [
    f"{TARGET_ZIMS_FOLDER}/{file}" for file in TARGET_ZIMS_FILES.split(",") 
]

def get_embedding(text, cache):
    text_key = text.strip()
    if not text_key: return None
    if text_key in cache: return cache[text_key]
    try:
        response = ollama.embeddings(model=EMBED_MODEL, prompt=text_key)
        embedding = response["embedding"]
        cache[text_key] = embedding
        return embedding
    except Exception:
        return None

def run_split_indexing():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    indexes_dir = os.path.join(script_dir, OUT_INDEX_FOLDER)
    os.makedirs(indexes_dir, exist_ok=True)

    print("=" * 75, flush=True)
    print("🚀 ЗАПУСК ПОЛНОЙ СЕПАРАТНОЙ ИНДЕКСАЦИИ ТЕХНИЧЕСКИХ БАЗ", flush=True)
    print(f"• Папка назначения для индексов: {indexes_dir}", flush=True)
    print(f"• Всего файлов в очереди на скан: {len(TARGET_ZIMS)} шт.", flush=True)
    print("=" * 75, flush=True)

    for idx, zim_path in enumerate(TARGET_ZIMS):
        if not os.path.exists(zim_path):
            print(f"⏩ [{idx+1}/{len(TARGET_ZIMS)}] Файл не найден: {zim_path}. Пропускаю.", flush=True)
            continue

        pure_filename = os.path.basename(zim_path)
        clean_name = pure_filename.replace("-", "_").replace(".", "_")
        cache_path = os.path.join(indexes_dir, f"cache_{clean_name}")
        
        print(f"\n📦 [{idx+1}/{len(TARGET_ZIMS)}] Скан архива: {pure_filename}", flush=True)
        print(f"   💾 Локальный кэш-файл: indexes/cache_{clean_name}.db", flush=True)
        
        start_time = time.time()
        added_count = 0
        skipped_count = 0
        
        try:
            archive = Archive(zim_path) # type: ignore
            searcher = Searcher(archive) 
            
            # Собираем уникальные пути со всей базы посимвольно
            unique_paths = set()
            print("   ⏳ Сбор структуры документов по буквенному индексу...", flush=True)
            
            # Пробегаемся по английскому алфавиту и цифрам
            search_symbols = list(string.ascii_lowercase) + [str(n) for n in range(10)]
            
            for symbol in search_symbols:
                try:
                    query = Query().set_query(symbol)
                    search = searcher.search(query)
                    # Выкачиваем до 20 000 статей на каждую букву
                    for path in search.getResults(0, 20000):
                        unique_paths.add(path)
                except Exception:
                    continue
            
            paths = list(unique_paths)
            total_paths = len(paths)
            print(f"   📊 Всего уникальных путей к статьям обнаружено: {total_paths}", flush=True)
            
            if total_paths == 0:
                print("   ⚠️ Не удалось извлечь структуру. Перехожу к следующему файлу.", flush=True)
                continue
            
            with shelve.open(cache_path) as cache:
                for path_idx, path in enumerate(paths):
                    try:
                        entry = archive.get_entry_by_path(path) 
                        
                        if any(entry.path.lower().endswith(ext) for ext in ['.css', '.js', '.png', '.jpg', '.svg', '.woff']):
                            continue
                            
                        raw_bytes = bytes(entry.get_item().content)
                        html_text = raw_bytes.decode('utf-8', errors='ignore')
                        
                        soup = BeautifulSoup(html_text, 'html.parser')
                        for element in soup(["script", "style", "nav", "footer", "header", "table"]):
                            element.extract()
                            
                        body_text = " ".join([line.strip() for line in soup.get_text().splitlines() if line.strip()])
                        if len(body_text) < 150: 
                            continue
                        
                        chunk = body_text[:1500]
                        if chunk.strip() in cache:
                            skipped_count += 1
                            continue
                        
                        if get_embedding(chunk, cache):
                            added_count += 1
                            
                        total_processed = added_count + skipped_count
                        if total_processed % 10 == 0 and total_processed > 0:
                            pct = (path_idx / total_paths) * 100
                            elapsed = time.time() - start_time
                            speed = total_processed / elapsed if elapsed > 0 else 1
                            eta = (total_paths - path_idx) / (path_idx / elapsed) if path_idx > 0 else 0
                            print(f"   🔹 Прогресс файла: {pct:.1f}% | Добавлено: {added_count} | Пропущено кэшем: {skipped_count} | Осталось времени: {eta:.0f} сек.", flush=True)
                            
                    except Exception:
                        continue
                        
            print(f"   🎉 Успешно завершено! База {pure_filename} готова.", flush=True)
            print(f"   📊 Итог файла -> Добавлено новых векторов: {added_count} | Пропущено (были в кэше): {skipped_count}", flush=True)
            print(f"   ⏱ Время обработки файла: {time.time() - start_time:.2f} сек.", flush=True)

        except Exception as e:
            print(f"   ❌ Критическая ошибка при обработке {pure_filename}: {e}", flush=True)

    print("\n" + "=" * 75, flush=True)
    print("✅ ВСЕ ВАШИ ТЕХНИЧЕСКИЕ БАЗЫ ДАННЫХ УСПЕШНО ПРОИНДЕКСИРОВАНЫ СЕПАРАТНО!", flush=True)
    print("=" * 75, flush=True)

if __name__ == '__main__':
    run_split_indexing()
