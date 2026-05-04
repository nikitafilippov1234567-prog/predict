import re
import requests
import json
from django.core.cache import cache
from django.conf import settings

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("⚠️  BeautifulSoup не установлен. Установите: pip install beautifulsoup4 lxml")


class DistrictAnalyzer:

    BLACKLIST_DISTRICTS = [
        'люблино', 'капотня', 'текстильщики', 'марьино', 'бирюлево',
        'некрасовка', 'выхино', 'кузьминки'
    ]

    WHITELIST_DISTRICTS = [
        'хамовники', 'пресненский', 'арбат', 'тверской', 'замоскворечье'
    ]

    # Сколько символов текста берём с каждой страницы при фетчинге
    PAGE_FETCH_CHARS = 4000
    # Топ-N URL из Tavily, которые фетчим полностью
    PAGE_FETCH_TOP_N = 5
    # Я люблю хардкод
    def __init__(self):
        self.openrouter_key = getattr(settings, 'OPENROUTER_KEY', 'your-key')
        self.tavily_key     = getattr(settings, 'TAVILY_KEY', 'your-key')
        self.enabled        = getattr(settings, 'DISTRICT_ANALYSIS_ENABLED', True)

    # ──────────────────────────────────────────────────────────────────────────
    # Публичный метод
    # ──────────────────────────────────────────────────────────────────────────

    def analyze_district(self, district, lat=None, lon=None):
        """Главный метод анализа района."""

        if not self.enabled or not district:
            return None

        cache_key = f'district_analysis_{district.lower()}'
        cached = cache.get(cache_key)
        if cached:
            print(f"✅ Анализ района '{district}' из кэша")
            return cached

        quick_result = self._quick_assessment(district)
        if quick_result:
            cache.set(cache_key, quick_result, 86400 * 30)
            print(f"✅ Быстрая оценка для района '{district}'")
            return quick_result

        try:
            print(f"🔍 Начинаем полный анализ района '{district}'")

            # 1. Поиск через Tavily (заголовки + сниппеты + URL)
            search_results = self._search_tavily(district) if self.tavily_key else []

            # 2. Фетчинг полного текста топ-N страниц
            rich_context = self._fetch_pages(search_results)

            # 3. Двухэтапный анализ через LLM с самопроверкой
            analysis = self._analyze_with_llm(district, rich_context, lat, lon) \
                       if self.openrouter_key else None

            if analysis:
                cache.set(cache_key, analysis, 86400 * 30)
                print(f"✅ Полный анализ района '{district}' завершён")
                return analysis
            else:
                return self._fallback_response(district)

        except Exception as e:
            print(f"❌ Ошибка анализа района '{district}': {e}")
            return self._fallback_response(district)

    # ──────────────────────────────────────────────────────────────────────────
    # Быстрая оценка по спискам
    # ──────────────────────────────────────────────────────────────────────────

    def _quick_assessment(self, district):
        d = district.lower()

        if any(bad in d for bad in self.BLACKLIST_DISTRICTS):
            return {
                'infrastructure_score': 3,
                'investment_score': 2,
                'ecology_score': 3,
                'overall_score': 2.7,
                'degradation_risk': 'высокий',
                'price_forecast_20y': 'снижение',
                'price_change_percent': '-15% до -30%',
                'red_flags': [
                    'Район в зоне риска деградации',
                    'Проблемы с экологией и инфраструктурой',
                    'Высокая доля арендного жилья',
                ],
                'green_flags': ['Относительно доступные цены'],
                'verdict': 'Район находится в процессе деградации. Не рекомендуется для долгосрочных инвестиций.',
                'recommendation': 'избегать',
            }

        if any(good in d for good in self.WHITELIST_DISTRICTS):
            return {
                'infrastructure_score': 9,
                'investment_score': 9,
                'ecology_score': 8,
                'overall_score': 8.7,
                'degradation_risk': 'низкий',
                'price_forecast_20y': 'значительный_рост',
                'price_change_percent': '+40% до +80%',
                'red_flags': ['Очень высокие цены'],
                'green_flags': [
                    'Престижная локация',
                    'Отличная инфраструктура',
                    'Центральное расположение',
                ],
                'verdict': 'Престижный район с отличными перспективами роста стоимости недвижимости.',
                'recommendation': 'для_инвестиций',
            }

        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Tavily: поиск
    # ──────────────────────────────────────────────────────────────────────────
    def _get_nominatim_context(self, district, lat, lon):
        """Получает административный контекст через Nominatim."""
        try:
            if lat and lon:
                url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&accept-language=ru"
            else:
                url = f"https://nominatim.openstreetmap.org/search?q={district}+Москва&format=json&limit=1&accept-language=ru"
            
            resp = requests.get(url, timeout=5, headers={'User-Agent': 'DistrictAnalyzer/1.0'})
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    data = data[0] if data else {}
                addr = data.get('address', {})
                return {
                    'suburb':   addr.get('suburb', ''),
                    'village':  addr.get('village', ''),
                    'city_district': addr.get('city_district', ''),
                    'county':   addr.get('county', ''),
                }
        except Exception as e:
            print(f"  ⚠️  Nominatim ошибка: {e}")
        return {}

    def _search_tavily(self, district, lat=None, lon=None):
        nom = self._get_nominatim_context(district, lat, lon)
        
        # Берём наиболее конкретный топоним из Nominatim
        settlement = (
            nom.get('suburb') or
            nom.get('village') or
            nom.get('city_district') or
            district
        )
        
        queries = [
            f"{district} {settlement} Москва отзывы жители инфраструктура 2024 2025",
            f"{settlement} новостройки цены недвижимость Москва",
            f"{settlement} Москва экология парки проблемы инфраструктура",
        ]

        all_results = []

        for query in queries:
            try:
                resp = requests.post(
                    'https://api.tavily.com/search',
                    json={
                        'api_key': self.tavily_key,
                        'query': query,
                        'max_results': 3,
                        'search_depth': 'basic',
                        'include_raw_content': False,   # полный текст берём сами через фетч
                    },
                    timeout=10,
                )
                if resp.status_code == 200:
                    results = resp.json().get('results', [])
                    all_results.extend(results)
                    print(f"  📝 Tavily [{len(results)}]: «{query[:55]}…»")
                else:
                    print(f"  ⚠️  Tavily HTTP {resp.status_code} для «{query[:55]}…»")
            except Exception as e:
                print(f"  ⚠️  Tavily ошибка: {e}")

        # Дедупликация по URL
        seen, unique = set(), []
        for r in all_results:
            url = r.get('url', '')
            if url and url not in seen:
                seen.add(url)
                unique.append(r)

        return unique

    # ──────────────────────────────────────────────────────────────────────────
    # Фетчинг полного текста страниц
    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_pages(self, search_results):
        """
        Берёт топ-N URL из результатов Tavily и скачивает полный текст.
        Возвращает список dict: {title, url, snippet, full_text}.
        """

        rich = []

        for r in search_results:
            snippet   = r.get('content', '')
            title     = r.get('title', 'Без названия')
            url       = r.get('url', '')
            full_text = ''

            if url and len([x for x in rich if x['full_text']]) < self.PAGE_FETCH_TOP_N:
                full_text = self._fetch_url_text(url)

            rich.append({
                'title':     title,
                'url':       url,
                'snippet':   snippet,
                'full_text': full_text,
            })

        return rich

    def _fetch_url_text(self, url):
        """Скачивает страницу и извлекает чистый текст."""

        skip_domains = ('youtube.com', 'youtu.be', 't.me', 'vk.com',
                        'instagram.com', 'facebook.com', 'twitter.com', 'x.com')
        if any(d in url for d in skip_domains):
            print(f"    ⏭  Пропускаем {url[:60]} (соцсеть/видео)")
            return ''

        try:
            resp = requests.get(
                url,
                timeout=8,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; DistrictBot/1.0)'},
            )
            resp.raise_for_status()

            if BS4_AVAILABLE:
                soup = BeautifulSoup(resp.text, 'lxml')
                for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                    tag.decompose()
                text = soup.get_text(separator=' ', strip=True)
            else:
                text = re.sub(r'<[^>]+>', ' ', resp.text)

            text = re.sub(r'\s+', ' ', text).strip()
            text = text[:self.PAGE_FETCH_CHARS]

            print(f"    🌐 Получено {len(text)} символов с {url[:55]}…")
            return text

        except Exception as e:
            print(f"    ⚠️  Не удалось скачать {url[:55]}…: {e}")
            return ''

    # ──────────────────────────────────────────────────────────────────────────
    # Двухэтапный LLM-анализ с самопроверкой
    # ──────────────────────────────────────────────────────────────────────────

    def _analyze_with_llm(self, district, rich_context, lat, lon):
        """
        Этап 1: первичный анализ на основе найденных материалов.
        Этап 2: самопроверка — модель критикует черновик и выдаёт финальный JSON.
        """

        context_blocks = []
        for i, r in enumerate(rich_context[:8]):
            header = f"=== Источник {i+1}: {r['title']} ===\nURL: {r['url']}"
            body   = r['full_text'] if r['full_text'] else r['snippet']
            if not body:
                continue
            context_blocks.append(f"{header}\n\n{body[:self.PAGE_FETCH_CHARS]}")

        context_text = "\n\n".join(context_blocks) if context_blocks \
                       else "Данные из поиска недоступны — анализируй только по общим знаниям."

        print("\n" + "─" * 60)
        print(f"  📚 КОНТЕКСТ ДЛЯ LLM ({len(context_blocks)} источников, ~{len(context_text)} символов):")
        print("─" * 60)
        for i, blk in enumerate(context_blocks):
            first_line = blk.split("\n")[0]
            print(f"  [{i+1}] {first_line[:100]}")
        print("─" * 60 + "\n")

        # ── Этап 1: первичный анализ (свободный текст) ───────────────────────
        prompt_draft = f"""Ты — аналитик рынка недвижимости Москвы с глубокими знаниями
о районах города. Твоя задача — оценить факторы спроса и цены жилья в районе
«{district}» (координаты: {lat}, {lon}).

## ИСТОЧНИКИ ИЗ ИНТЕРНЕТА
{context_text}

## ПРАВИЛА РАБОТЫ С ИСТОЧНИКАМИ

Для каждого найденного факта из источника пиши:
[Источник N]: [пересказ факта] → влияние на цену: рост / нейтрально / снижение

Если источники по фактору пусты или содержат только технические данные (CSS,
капча, скрипты) — используй собственные знания о районе и пиши:
[base_knowledge]: [факт] → влияние: ...

## ЧТО АНАЛИЗИРОВАТЬ

### A. Локационные факторы (вес: высокий)
— Метро/МЦК: есть ли станция, время до центра (Кольцевая/Сити).
— Шоссе и выезды из района.
— Принадлежность к округу (ЦАО/ЗАО/НАО/ТАО и т.д.) и удалённость от МКАД.

### B. Рыночные факторы (вес: высокий)
— Ценовой диапазон новостроек и вторичного рынка (₽/м²).
— Классы жилья, присутствующие в районе.
— Активность застройщиков: количество проектов, темп продаж.
— Редевелопмент, программа реновации.

### C. Средовые факторы (вес: средний)
— Парки, водоёмы, зелёные зоны рядом.
— Промзоны, ТЭЦ, шумовые зоны — только если реально присутствуют.
— Качество существующей застройки (хрущёвки / панель / монолит / новостройки).

### D. Риски снижения стоимости (вес: высокий)
— Избыток предложения (оверсаплай новостроек).
— Социальные проблемы, высокая аварийность жилья.
— Планируемые негативные изменения (трассы, промобъекты).

## ВАЖНО
Не пиши «данных нет» по факторам A и B для района Москвы — ты знаешь
московский рынок. Используй [base_knowledge] вместо пробела."""

        draft = self._llm_call(prompt_draft, max_tokens=1800, label="Этап 1 (черновик)")
        if not draft:
            return None

        print("\n" + "─" * 60)
        print("  📄 ЧЕРНОВИК (Этап 1):")
        print("─" * 60)
        print(draft)
        print("─" * 60 + "\n")

        # ── Этап 2: самопроверка + финальный JSON ────────────────────────────
        prompt_final = f"""Ты — строгий экономический редактор анализа рынка недвижимости Москвы.

## ЧЕРНОВОЙ АНАЛИЗ РАЙОНА «{district}»
{draft}

## ШАГ 1. КРИТИКА

А) Проверь: есть ли выводы «рост цен» основанные только на факте наличия
   новостроек или экопарка — без данных о спросе? Пометь как [слабый сигнал].

Б) Проверь баланс: если по факторам A и B преобладают позитивные сигналы,
   а в D рисков нет — recommendation не может быть «избегать».

В) Проверь confidence: если большинство фактов из [base_knowledge] — это
   «средняя», не «низкая». LLM знает московский рынок.

## ШАГ 2. ФИНАЛЬНЫЙ JSON

Только JSON-объект, без текста до и после.

Поля:
- demand_pressure: «высокий» / «средний» / «низкий»
- supply_trend: «рост» / «стагнация» / «снижение»
- degradation_risk: «высокий» / «средний» / «низкий»
- overall_score: число от 1 до 10 (взвешенная оценка инвестпривлекательности)
- confidence: «высокая» / «средняя» / «низкая»
- green_flags: 2–4 подтверждённых позитивных фактора, каждый 1–2 предложения
- red_flags: подтверждённые риски, каждый 1–2 предложения (пустой список допустим)
- location_summary: 3–5 предложений о транспортной доступности, удалённости
  от центра, принадлежности к округу и ключевых магистралях
- market_summary: 3–5 предложений о ценовом диапазоне, классах жилья,
  активности застройщиков, ликвидности рынка
- risk_summary: 3–5 предложений об основных рисках района — оверсаплай,
  социальные проблемы, инфраструктурный дефицит. Если рисков нет — напиши об этом
- verdict: итоговый вывод для покупателя, 4–6 предложений: баланс плюсов
  и минусов, для кого подходит район, горизонт владения

Логика overall_score:
- Локация (A) + рынок (B) весят 60%, среда (C) + риски (D) весят 40%
- overall_score ≥ 7 → рекомендация «для_инвестиций»
- overall_score 5–6 → «для_жизни»
- overall_score ≤ 4 → «избегать»

{{
  "demand_pressure": "...",
  "supply_trend": "...",
  "degradation_risk": "...",
  "overall_score": ...,
  "confidence": "...",
  "green_flags": ["...", "..."],
  "red_flags": ["...", "..."],
  "location_summary": "...",
  "market_summary": "...",
  "risk_summary": "...",
  "verdict": "..."
}}"""

        final_text = self._llm_call(prompt_final, max_tokens=1000, label="Этап 2 (проверка+JSON)")
        if not final_text:
            return None

        print("\n" + "─" * 60)
        print("  🔍 САМОПРОВЕРКА + JSON (Этап 2):")
        print("─" * 60)
        print(final_text)
        print("─" * 60 + "\n")

        return self._extract_json(final_text)

    # ──────────────────────────────────────────────────────────────────────────
    # Вспомогательные методы LLM
    # ──────────────────────────────────────────────────────────────────────────

    def _llm_call(self, prompt, max_tokens=1000, label="LLM"):
        """Один вызов OpenRouter. Возвращает текст ответа или None."""

        try:
            resp = requests.post(
                'https://openrouter.ai/api/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.openrouter_key}',
                    'Content-Type': 'application/json',
                    'HTTP-Referer': 'https://yourdomain.com',
                    'X-Title': 'Real Estate Analyzer',
                },
                json={
                    'model': 'deepseek/deepseek-v3.2',
                    'provider': {'only': ['atlas-cloud/fp8']},
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0.2,    # ниже температура → меньше галлюцинаций
                    'max_tokens': max_tokens,
                },
                timeout=45,
            )

            if resp.status_code == 200:
                content = resp.json()['choices'][0]['message']['content']
                print(f"  ✅ {label}: {len(content)} символов")
                return content
            else:
                print(f"  ❌ {label} HTTP {resp.status_code}: {resp.text[:200]}")

        except Exception as e:
            print(f"  ❌ {label} ошибка: {e}")

        return None

    def _extract_json(self, text):
        """Извлекает первый валидный JSON-объект из текста."""

        j0 = text.find('{')
        j1 = text.rfind('}') + 1
        if j0 == -1 or j1 <= j0:
            print(f"  ⚠️  JSON не найден в ответе:\n{text[:300]}")
            return None
        try:
            parsed = json.loads(text[j0:j1])
            print(f"  ✅ JSON разобран: overall_score={parsed.get('overall_score')}")
            return parsed
        except json.JSONDecodeError as e:
            print(f"  ⚠️  Ошибка парсинга JSON: {e}\n{text[j0:j1][:300]}")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # Заглушка
    # ──────────────────────────────────────────────────────────────────────────

    def _fallback_response(self, district):
        return {
            'infrastructure_score': 5,
            'investment_score': 5,
            'ecology_score': 5,
            'overall_score': 5.0,
            'degradation_risk': 'средний',
            'price_forecast_20y': 'стагнация',
            'price_change_percent': '-5% до +15%',
            'red_flags': ['Недостаточно данных для анализа'],
            'green_flags': ['Требуется дополнительное исследование'],
            'verdict': f'Для района {district} недостаточно данных. Рекомендуем изучить локацию лично.',
            'recommendation': 'для_жизни',
        }


# ══════════════════════════════════════════════════════════════════════════════
# Автономное тестирование:  python district_analyzer.py
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import os
    import sys

    # ── Django-заглушка ───────────────────────────────────────────────────────
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', '')
    try:
        import django
        from django.conf import settings as _s
        if not _s.configured:
            _s.configure(
                DISTRICT_ANALYSIS_ENABLED=True,
                CACHES={'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}},
            )
            django.setup()
    except Exception as e:
        print(f'⚠️  Django init: {e}')
        sys.exit(1)

    # ── Тестовый подкласс (ключи без Django settings) ─────────────────────────
    class _TestAnalyzer(DistrictAnalyzer):
        def __init__(self):
            self.openrouter_key = os.getenv(
                'OPENROUTER_KEY',
                'your-key(тот же)'
            )
            self.tavily_key = os.getenv(
                'TAVILY_KEY',
                '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'
            )
            self.enabled = True

    # ── Тест-кейсы ────────────────────────────────────────────────────────────
    TEST_CASES = [
        ('Хамовники', 55.7279, 37.5765, 'для_инвестиций'),   # whitelist → quick
        ('Капотня',   55.6478, 37.7825, 'избегать'),           # blacklist → quick
        ('Раменки',   55.7100, 37.4200,  None),                # LLM + фетчинг + самопроверка
    ]

    REQUIRED_KEYS = [
        'infrastructure_score', 'investment_score', 'ecology_score',
        'overall_score', 'degradation_risk', 'price_forecast_20y',
        'price_change_percent', 'red_flags', 'green_flags',
        'verdict', 'recommendation',
    ]

    print('=' * 66)
    print('  ТЕСТИРОВАНИЕ DistrictAnalyzer  (deepseek-v3.2 + самопроверка)')
    print('=' * 66)

    analyzer = _TestAnalyzer()
    all_ok   = True

    for district, lat, lon, expected_rec in TEST_CASES:
        print(f'\n🏙  Район: {district}  [{lat}, {lon}]')
        result = analyzer.analyze_district(district, lat=lat, lon=lon)

        if result is None:
            print('  ❌ Получен None — анализ не выполнен')
            all_ok = False
            continue

        missing = [k for k in REQUIRED_KEYS if k not in result]
        if missing:
            print(f'  ❌ Отсутствуют ключи: {missing}')
            all_ok = False
        else:
            print('  ✅ Все обязательные ключи присутствуют')

        for sk in ('infrastructure_score', 'investment_score', 'ecology_score', 'overall_score'):
            v = result.get(sk)
            if not (isinstance(v, (int, float)) and 1 <= v <= 10):
                print(f'  ⚠️  {sk} = {v!r} — вне диапазона 1–10')
                all_ok = False

        actual_rec = result.get('recommendation')
        if expected_rec:
            mark = '✅' if actual_rec == expected_rec else '⚠️ '
            extra = '' if actual_rec == expected_rec else f', ожидалось {expected_rec!r}'
            print(f'  {mark} recommendation = {actual_rec!r}{extra}')

        print(f'  📊 инфра={result.get("infrastructure_score")}  '
              f'инвест={result.get("investment_score")}  '
              f'эко={result.get("ecology_score")}  '
              f'итого={result.get("overall_score")}')
        print(f'  🏷  Риск деградации : {result.get("degradation_risk")}')
        print(f'  📈 Прогноз 20 лет  : {result.get("price_forecast_20y")} '
              f'({result.get("price_change_percent")})')
        print(f'  💬 Вердикт         : {result.get("verdict")}')

    print('\n' + '=' * 66)
    print(f'  ИТОГ: {"✅ ВСЕ ТЕСТЫ ПРОШЛИ" if all_ok else "❌ ЕСТЬ ПРОБЛЕМЫ — см. выше"}')
    print('=' * 66)
    sys.exit(0 if all_ok else 1)
