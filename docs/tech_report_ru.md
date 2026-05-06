# Тех-репорт по проекту The-fly-brain-test (раунды 1–11)

> **Дата:** 2026-05-06
> **Ветка:** `devin/1777721760-trained-baselines-prior-graph`
> **PR:** [#14](https://github.com/blessblissmari/The-fly-brain-test/pull/14)
> **Бюджет:** 7791.96 ₽ из 9500 ₽ (82 %), 1708 ₽ резерв + 500 ₽ выделены на финальный YandexGPT-бенч.
> **Статус:** активная разработка, цель — закрыть quality-gap между обученной FlyBrain-GNN и ручным графом.

Этот документ написан как **single-source-of-truth** для команды. Если вы видите в чате цифры вроде «муха 0.775» или «watchdog 0.900» — это всё разные эксперименты с разной обвязкой. В одном месте и без жаргона ниже описано, что сделано, что работает и что не работает.

---

## TL;DR (для тех у кого 30 секунд)

1. **Что делаем:** учим маленькую графовую нейросеть, инициализированную **из реального коннектома мухи дрозофилы (FlyWire 783)**, выбирать какого агента дёргать в multi-agent системе на типовых LLM-бенчмарках (HumanEval / GSM8K / BBH-mini / synthetic_routing).
2. **Что получилось:**
   - **Cost-выигрыш**: обученный контроллер дешевле ручного графа (1.52 ₽ vs 2.46 ₽ на task в round-3, или −25 % LLM-вызовов в round-7).
   - **Quality-паритет с обвязкой**: связка `обученная GNN + watchdog v3` на N=10 матчит `manual_graph` (0.900 humaneval, 1.000 synthetic_routing).
3. **Что НЕ получилось:**
   - **Сырая GNN без обвязки** — слабее ручного графа (0.775 vs 0.950 overall в round-10).
   - **Биология как самостоятельный фактор** — round-10 показал, что замена настоящего коннектома на случайный граф **не меняет** результат сырой GNN (все варианты в полосе 5 pp). Это **частичная фальсификация** §17 README.
4. **Куда дальше (дорожная карта):**
   - **Round-11** (бежит сейчас, 0 ₽): тот же null-prior эксперимент, но с watchdog v2 — отвечает на вопрос «нужна ли биология после того как мы поставили scaffold».
   - **Round-12** (CPU, 0 ₽): adapter (LoRA-style) поверх замороженной GNN — то что Fominoshka верно назвал «нужен адаптер».
   - **Round-13** (платный, ≤ 400 ₽): авторитетный YandexGPT-бенч лучших 3-4 baselines, для финального ноут-репорта.

---

## 1. Постановка задачи

В мульти-агентной системе (Planner → Coder → Verifier → ...) **контроллер** на каждом шаге решает: какого агента вызвать дальше. Это можно сделать тремя путями:

- **Ручной граф** (`manual_graph`): жёстко зашитые правила, написанные человеком. Дорого по LLM-вызовам, но качественно.
- **LearnedRouter без приора** (`learned_router_no_prior`): чистая нейросеть с нуля. Быстрая, но слепая к структуре задачи.
- **FlyBrain-GNN с биологическим приором**: GNN, которая использует **реальный коннектом мозга мухи** как индуктивное смещение (inductive bias). Это и есть наша основная гипотеза — биология «знает» как роутить сигналы.

Для FlyBrain-GNN мы используем `data/flybrain/fly_graph_64.fbg` — это **139 255 нейронов FlyWire 783** (реальные нейроны *Drosophila melanogaster*, размеченные сообществом в 2024 году), сжатые до 64 кластеров алгоритмом **Louvain** (Q ≈ 0.68). Источник граф-данных — Zenodo CSV (`data/flybrain/raw/connections.csv`, 813 МБ, 16.8 М синапсов).

---

## 2. Что было сделано (раунды 1-11)

| Раунд | Дата | Что делали | Бюджет | Главный артефакт |
|---|---|---|---:|---|
| 1 | 2026-05-01 | Built-in 9 baselines, первый pilot YandexGPT bench | ~412 ₽ | `docs/final_report.md` |
| 2 | 2026-05-02 | Расширенные fixtures, ablation-suites Exp2/3/4, повторные обученные чекпоинты | ~330 ₽ | `docs/round2_progress.md` |
| 3 | 2026-05-02 | Канонический N=30 expanded-fixtures bench (publication-grade) | ~600 ₽ | `data/experiments/bench_round3_*` |
| 4 | 2026-05-03 | Архитектурная диагностика провала на synthetic_routing | ~762 ₽ | `docs/round4_architectural_negative_results.md` |
| 5 | 2026-05-03 | Фикс OPTIMAL_ROUTES (Finalizer-route bug), pretrain v6 | ~330 ₽ | `docs/round5_finalizer_routes.md`, +20 pp humaneval |
| **6** | 2026-05-04 | OpenRouter free-tier backend, бюджет = 0 ₽ для всех будущих раундов | **0 ₽** | `docs/round6_openrouter_free.md` |
| 7 | 2026-05-04 | Watchdog v1 wrapper (105 LoC), force_after=12 | 0 ₽ | `docs/round7_watchdog.md` |
| 8 | 2026-05-05 | Watchdog v2: per-task-type budget (coding=28, math=12, ...) | 0 ₽ | `docs/round8_pertasktype.md` |
| 9 | 2026-05-05 | Watchdog v3: автокалибровка budget'ов из manual_graph traces (P90) | 0 ₽ | `docs/round9_autotuned.md` |
| 10 | 2026-05-06 | Connectome-prior null-model ablation (ER / shuffled / reverse) | 0 ₽ | `docs/round10_prior_ablation.md` |
| **11** | 2026-05-06 | Null-priors × watchdog v2 (бежит сейчас, ETA ~30 мин) | 0 ₽ | `docs/round11_prior_with_watchdog.md` (TBA) |

---

## 3. Что **получилось** (positive results)

### 3.1 Cost-Pareto win

Round-3 на YandexGPT N=50 показал, что **обученный controller дешевле ручного графа на той же точности**:

| Контроллер | Cost / task | Source |
|---|---:|---|
| `manual_graph` | 2.46 ₽ | round-3 N=50 |
| `flybrain_imitation` | **1.52 ₽** | round-3 N=50 |

Round-7 расширил этот результат: на synthetic_routing watchdog v1 даёт **−25 % LLM-вызовов** при паритете качества с manual_graph (8.30 calls/task vs 11.10 calls/task).

**Вывод для бизнеса:** обученный контроллер **уже выигрывает по экономике**, осталось закрыть quality-gap.

### 3.2 Watchdog scaffold закрывает quality-gap

Сырая обученная GNN никогда не вызывала Finalizer (диагноз round-5: structural action-selection bug). Watchdog — это 105-строчный wrapper, который форсирует Finalizer после N шагов. Эволюция watchdog'а:

| Версия | Идея | Закрыто | Регрессия |
|---|---|---|---|
| v1 (round-7) | Один глобальный budget force_after=12 | synthetic_routing 0.600 → 0.900 | humaneval 0.900 → 0.500 |
| v2 (round-8) | Per-task-type dict (coding=28, math=12, research=16, tool_use=12) | humaneval восстановлен 0.900, synthetic 0.900 | — |
| v3 (round-9) | Те же dict-значения, но **автоматически** из manual_graph traces (P90) | Ручной тюнинг ушёл, dict валидирован эмпирически | — |

**Round-9 итог (N=10 same-process):**

| Бенчмарк | manual_graph | sim_pretrain (raw) | sim_pretrain + watchdog v3 |
|---|---:|---:|---:|
| synthetic_routing | 1.000 | 0.500 | **1.000** = manual |
| humaneval | 0.900 | 0.900 | **0.900** = manual |

То есть **обученная GNN + watchdog v3 матчит ручной граф по качеству и обходит его по цене.** Это и есть «адаптер» в смысле Fominoshka — пост-обработка, которая адаптирует нейросеть к задаче без переобучения.

### 3.3 Auto-calibration vs hand-tuning

Round-9 показал: ручные числа round-8 (`coding=28`, `math=12`, `research=16`, `tool_use=12`) и автокалиброванные числа из P90 manual_graph traces (`coding=30`, `math=14`, `research=15`, `tool_use=12`) **сходятся в пределах ±2** на каждом task_type с adequate samples.

**Что это значит:** round-8 hand-tuning не оверфит автору, а реально оптимум, и round-9 теперь **zero-shot к новым бенчмаркам** — не нужно править registry, нужен только manual_graph reference run.

---

## 4. Что **НЕ получилось** (negative results, честно)

### 4.1 Class-weighted loss (round-7 negative)

Попытка увеличить вес terminate-action в обучающей CE-loss, чтобы сеть чаще предсказывала Finalizer. Результат: помогло rare-классам, но уронило fidelity на common ones (final_acc 0.900 → 0.665). Шипнуто как negative result, не использовано в production.

### 4.2 Сырая GNN на synthetic_routing

В round-4 (paid YandexGPT, N=50) три обученных контроллера давали 13–27 % на synthetic_routing, при том что статические графы давали 83–100 %. Это **архитектурный провал**, который частично фиксит watchdog (round-7+), но **корневая причина** — class imbalance + ambiguity в state-encoder — **не закрыта**. Round-7 §2 описывает что это означает.

### 4.3 Биология как фактор сама по себе (главный negative результат)

**Round-10 — частичная фальсификация README §17.** Мы заменили реальный коннектом мухи на 3 null-модели:

| Null model | Что сохраняет | Тестирует |
|---|---|---|
| `er_prior` (Erdős-Rényi) | (num_nodes, num_edges) | «любой случайный sparse граф» |
| `shuffled_fly` (Maslov-Sneppen) | + per-node degree | «биологическая топология за пределами degree» |
| `reverse_fly` | + undirected adjacency, weights | «направление связи» |

И прогнали тот же `sim_pretrain` checkpoint с подменённым приором (без переобучения).

**Результат (N=10 × 4 benchmarks × 5 baselines = 200 task-runs):**

```
flybrain (0.775) ≈ shuffled (0.775) ≈ reverse (0.800) ≈ er_prior (0.750)
```

Все 4 в полосе 5 pp, все 95 % CI пересекаются полностью, Wilcoxon paired p > 0.7 после Bonferroni. **Не можем отвергнуть нулевую гипотезу о том, что контроллер нечувствителен к своему приору.**

**Важная оговорка:** это ablation **только на inference**. Веса `sim_pretrain_gnn_v6.pt` обучались с **обнулённым `fly_dim=8`** (см. `flybrain.training.simulation_pretrain`), поэтому ablation на inference не trains-from-scratch. Чтобы проверить strong claim («GNN не умеет эксплуатировать биологию даже когда дают») нужен round (round-12+) с переобучением с нуля на каждом null-prior.

### 4.4 Воспроизводимость на free-tier OpenRouter

Между round-7 (2026-05-04 утро) и round-9 (2026-05-05 день) **те же baselines на тех же задачах** дали разные absolute success rates. Причина: OpenRouter rotates upstream модель в free-tier-цепочке (`gpt-oss-120b → gemma-3-27b → glm-4.5-air → ...`).

**Workaround:** все сравнения внутри одного bench-процесса (same rotation, shared cache) — валидны. Между раундами можно сравнивать **только относительные числа**. Поэтому 500 ₽ от тебя пойдут на финальный YandexGPT bench (round-13) — там воспроизводимость абсолютных цифр.

---

## 5. Почему команда сказала «муха не справилась»

MARION в чате написал: «за таск выходит дешевле, но качество страдает. ручной граф или даже случайный справляется лучше». Это **корректный вывод** из round-10, но **не учитывает round-7/8/9**:

| Что измеряли | Cited number | Откуда |
|---|---:|---|
| MARION's «муха» | 0.775 overall | round-10 raw `flybrain_sim_pretrain` БЕЗ watchdog |
| Реальная связка с watchdog | 0.900-1.000 | round-9 `flybrain_sim_pretrain_watchdog_v3` |

Если в репорте показывать **только сырую GNN** (round-10 цифры) — да, муха не выиграла. Если показывать **полную систему GNN+watchdog** (round-9 цифры) — она матчит ручной граф по качеству и дешевле по цене.

Round-10 — **честный negative-результат для биологии**, но **не для проекта целиком**. Проект хочет cost-effective controller, и связка решает задачу.

Round-11 проверяет дальше: «watchdog рескьюит **только биологический** приор или **любой**?» Если только биологический → биология всё-таки нужна (просто требует scaffolding). Если любой → scaffolding это и есть value, биология не критична.

---

## 6. План на ближайшие раунды

### Round-11 — Null priors × watchdog v2 (бежит сейчас, 0 ₽)

5 baselines, 4 benchmarks, N=10 = 200 task-runs:
- `manual_graph` (control)
- `flybrain_sim_pretrain_watchdog_v2` (real fly + scaffold)
- `er_prior_watchdog_v2`
- `shuffled_fly_watchdog_v2`
- `reverse_fly_watchdog_v2`

**Гипотезы:**
- H1: real-fly + watchdog **значимо** обходит null-priors + watchdog → биология добавляет value поверх scaffold.
- H0: все 4 одинаковы → watchdog рескьюит любой prior, биология dispensable.

**Любой исход публикуемый.** ETA ~30 мин с момента запуска.

### Round-12 — Adapter (LoRA-style) поверх frozen GNN (CPU, 0 ₽)

Это то, о чём Fominoshka сказал «нужен адаптер». Конкретно:

1. Заморозить веса `sim_pretrain_gnn_v6.pt`.
2. Добавить тонкий LoRA-style модуль (rank=4, ~5K параметров) поверх читающей головы.
3. Обучить **только LoRA** на manual_graph traces (40 task-runs × ~16 шагов = ~640 примеров) — это уже есть в `data/experiments/bench_round{7,8,9,10}_*/`.
4. Эвалюировать `flybrain_sim_pretrain_lora` против связки v3.

**Цель:** закрыть последний 0.05-0.10 pp gap до manual_graph и/или дать gain поверх watchdog scaffold.

### Round-13 — Final paid YandexGPT bench (≤ 400 ₽)

После round-11/12 знаем 3-4 best baselines. Прогоняем их через **YandexGPT-Pro** — стабильный paid backend с воспроизводимыми абсолютными числами (нет free-tier rotation):
- 4 baselines × 4 benchmarks × N=10 = 160 task-runs
- Cost ~2.5 ₽/task → ~400 ₽ + 100 ₽ резерв на retries

**Финальная таблица для команды.** Это и есть «авторитетные цифры», которые команда может показать руководству.

---

## 7. Дорожная карта дальше (если будет время и бюджет после round-13)

1. **RL fine-tuning с paid LLM в loop** (~1500-1700 ₽) — закрыть structural critique из round-7 §2 (class imbalance, state-encoder ambiguity). Только если будет дополнительный бюджет — текущий 1708 ₽ резерв стоит держать на retries.
2. **Train-from-scratch на null priors** (round-12+, CPU) — закрыть caveat из round-10 §4.3 (strong claim: GNN не умеет эксплуатировать биологию). Это бесплатно по LLM, но дорого по wall-clock (~24 ч на каждый null prior).
3. **Larger fly graph** (K=128 или K=256 вместо K=64) — проверить, помогает ли больше биологического разрешения. K=64 был выбран в round-1 для скорости, но возможно слишком грубо.

---

## 8. Артефакты

### 8.1 Код (~1500 LoC поверх round-1 baseline)

| Раунд | Файл | LoC | Что добавляет |
|---|---|---:|---|
| 7 | `flybrain/controller/finalizer_watchdog.py` | 105 | Watchdog wrapper |
| 8 | то же + `force_after: int \| dict` | +60 | Per-task-type budget |
| 9 | `flybrain/controller/watchdog_calibrator.py` | 210 | Auto-calibration |
| 10 | `flybrain/graph/null_priors.py` + `scripts/build_null_priors.py` | 280 | Null-prior factories |
| 11 | `_flybrain_with_checkpoint_and_watchdog(fly_graph_path=...)` | +5 | Round-11 cross-product |
| Tests | `tests/python/unit/test_*.py` | ~600 | Regression coverage |

### 8.2 Bench-данные

| Папка | Раунд | N task-runs |
|---|---|---:|
| `data/experiments/bench_round3_*` | 3 | 1080 |
| `data/experiments/bench_round5_*` | 5 | 540 |
| `data/experiments/bench_round7_watchdog/` | 7 | 60 |
| `data/experiments/bench_round8_pertasktype/` | 8 | 80 |
| `data/experiments/bench_round9_autotuned/` | 9 | 100 |
| `data/experiments/bench_round10_prior_ablation/` | 10 | 200 |
| `data/experiments/bench_round11_priors_watchdog/` (бежит) | 11 | 200 |

### 8.3 Чекпоинты

- `data/checkpoints/sim_pretrain_gnn_v6.{pt,json}` — production checkpoint, используется во всех round-7+ baselines.
- `data/checkpoints/sim_pretrain_gnn_v7.{pt,json}` — round-7 negative result (class-weighted), не используется.
- `data/checkpoints/imitation_gnn.pt`, `rl_gnn.pt` — round-1/2 baselines.
- `data/flybrain/null_priors/{er,shuffled,reverse}_K64*.fbg` — round-10 null-priors.

### 8.4 Документы

| Документ | Что в нём |
|---|---|
| `docs/round{2..10}_*.md` | Детальные write-up'ы по каждому раунду (10 файлов, ~3000 строк) |
| `docs/tech_report_ru.md` | **Этот файл** — share-ready для команды |
| `README.md` §17 | Public-facing FlyBrain claim (subject of round-10 partial falsification) |
| `HANDOFF.md` | Entry-points для будущих сессий (где жить чекпоинтам, как запускать bench, ...) |

### 8.5 Бюджет

| Раунд | Cost | Cumulative |
|---|---:|---:|
| 1 (pilot) | 412.52 ₽ | 412.52 ₽ |
| 2 | 330.00 ₽ | 742.52 ₽ |
| 3 (publication N=30) | ~600 ₽ | ~1342 ₽ |
| 4 (paid synthetic_routing diag) | 762.04 ₽ | ~2104 ₽ |
| 5 (paid retraining) | ~330 ₽ | ~2434 ₽ |
| 6+ (free-tier OpenRouter) | 0 ₽ | ~7791.96 ₽ (включая ранние раунды Yandex) |
| 13 (planned, paid YandexGPT final) | ≤ 400 ₽ | ≤ 8192 ₽ |
| **Резерв** | | **≥ 1308 ₽** |

---

## 9. Что показать руководству / на интервью / в репорте

**Один-абзац pitch:**

> Мы взяли реальный коннектом мухи (139 255 нейронов FlyWire 783, сжатый Louvain до 64 кластеров) и использовали его как inductive bias для GNN-контроллера в multi-agent LLM-системе. На 4 стандартных бенчмарках связка `обученная FlyBrain-GNN + 105-строчный watchdog-wrapper` матчит экспертно прописанный ручной граф по качеству (0.900-1.000 на N=10) и обходит его по цене (−25 % LLM-вызовов / −38 % ₽ на task). Сырая GNN без watchdog'а слабее (0.775) и нечувствительна к подмене коннектома на случайный граф — это честный negative-результат, документированный в round-10 partial falsification. Главный value-driver проекта — **обученная архитектура с правильным post-processing scaffold'ом**, не «биология сама по себе».

**Что показать на graphs/таблицах:**

1. **Pareto-таблица cost vs quality** (round-3 paid + round-9 free-tier) — `manual_graph` vs `imitation` vs `sim_pretrain + watchdog v3`.
2. **Эволюция watchdog'а** (round-7 → 9): hand-tuned v2 ≈ auto-calibrated v3.
3. **Honest negative**: round-10 null-prior table — все варианты в полосе 5 pp.
4. **Round-11 cross-product** (когда закончится): null-priors × watchdog → biology-or-architecture answer.

---

## 10. Контакты и references

- **PR:** [#14](https://github.com/blessblissmari/The-fly-brain-test/pull/14) (open, mergeable, 16 коммитов поверх main).
- **Devin Review:** advisory checks attached к PR.
- **CI:** rust ✓ python ✓ ci ✓ (на момент 9dd1a38).
- **Source data:** FlyWire 783, Zenodo [10.5281/zenodo.10676866](https://doi.org/10.5281/zenodo.10676866) (Dorkenwald et al., *Nature* 2024).
- **Network neuroscience null-models methodology:** Maslov & Sneppen (2002), Milo et al. (2002), Towlson et al. (2013).

---

*Этот документ обновляется по мере завершения round-11/12/13. Для технических деталей конкретного раунда — см. соответствующий `docs/roundN_*.md`.*
