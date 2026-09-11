# Claude Code gegen den lokalen Server: Diagnose und Behebung

**Untersucht:** 2026-09-09, ergänzt 2026-09-10 und 2026-09-11 · **Maschine:** x99, gfx906 —
**16 GB + 32 GB, asymmetrisch** (Device 0 MI50/MI60 16 368 MiB, Device 1 32 752 MiB) ·
**Stack:** llama-swap → mx-org-Build, Layer-Split mit `--tensor-split 5,1`

Claude Code lief gegen den lokalen `ornith`-Endpunkt spürbar zäh: lange Wartezeiten, abgebrochene
Werkzeugaufrufe, blockierte Bash-Befehle. Die Ursache war **nicht** die Hardware — Prefill und
Decode liegen auf dieser Maschine bei ~1000 bzw. ~55 t/s. Es waren **sechs** voneinander
unabhängige Probleme in der Konfiguration und im Protokollpfad — das sechste (Befund 6, ein
stummes 404 auf Hintergrund-Anfragen) kam erst im echten Betrieb am Folgetag zum Vorschein.

---

## Aufbau

`~/.local/bin/claude-local` setzt `ANTHROPIC_BASE_URL=http://192.168.178.71:8080` sowie
`ANTHROPIC_MODEL` und `ANTHROPIC_SMALL_FAST_MODEL` (beide auf `${CLAUDE_LOCAL_MODEL:-claude}`).
Claude Code spricht damit die **Anthropic-Messages-API** (`/v1/messages`), die llama.cpp emuliert.
Dahinter steht llama-swap mit acht Modellen in einer `exclusive`-Gruppe.

Ursprünglich zeigte das Skript auf `ornith` und setzte nur `ANTHROPIC_MODEL` — beides ist
inzwischen geändert (Endstand bzw. Befund 6).

---

## Befund 1: Das Denk-Budget fraß die Antworten

`qwen_args` enthielt `--reasoning-budget 16000`. Der Hilfetext: *„token budget for thinking"*.

Gemessen mit einer 39-Token-Frage und `max_tokens: 200`:

| | |
|---|---|
| Antwortinhalt | **leer** |
| Reasoning | 813 Zeichen |
| Completion | 200 Tokens — Limit erreicht |
| Dauer | **66 s** |

Das Nachdenken verbrauchte das gesamte Token-Limit, bevor Inhalt entstand. Bei einem Agenten, der
pro Aufgabe zwanzig kleine Werkzeugaufrufe macht, ist das der Normalfall, nicht die Ausnahme.

Selbst für **„Sage genau: OK"** erzeugte das Modell 194 Zeichen Denkspur vor der Antwort.

**Rechnung:** 16 000 Tokens ÷ 55 t/s ≈ **290 s** pro Aufruf im Extremfall.

## Befund 2: Denkspuren wuchsen in den Kontext

`--reasoning-preserve` war aktiv:

> *preserve reasoning trace in the **full history**, not just the last assistant message*

Jede Denkphase blieb dauerhaft im Verlauf. Nach zehn Werkzeugaufrufen trägt man leicht 50 000
Tokens angesammelter Gedanken mit, die bei jedem weiteren Aufruf erneut durch den Prefill müssen.

**Wichtiger Nebeneffekt:** Ein wachsendes Präfix entwertet den Prompt-Cache (siehe Befund 5).

## Befund 3: Der Permission-Classifier blockierte Bash

Claude Code fragt im Auto-Modus vor jedem Bash-Befehl ein Modell, ob der Befehl sicher ist. Als
Classifier diente `ornith` selbst. Meldung bei Zeitüberschreitung:

```
ornith is temporarily unavailable (timed out), so auto mode cannot determine
the safety of Bash right now.
```

Zwei Auslöser:

- **Kaltstart.** `ttl: 3600` entlädt das Modell nach einer Stunde; der nächste Classifier-Aufruf
  löst ein Nachladen von 27 GB aus. `healthCheckTimeout: 300` lässt llama-swap fünf Minuten warten
  — Claude Codes eigenes Zeitfenster ist viel kürzer.
- **Verdrängung.** Die `exclusive`-Gruppe erlaubt nur ein geladenes Modell; jede Anfrage an ein
  anderes verdrängt `ornith`.

## Befund 4: `output_config` wird nicht unterstützt

Claude Code schickt auf der Anthropic-Route strukturierte Ausgaben mit:

```json
"output_config": { "effort": "high", "format": { "type": "json_schema", "schema": {…} } }
```

`grep -rn output_config tools/server/` im mx-org-Build: **null Treffer.** Das Feld wird ignoriert.

Ergebnis über `/v1/messages` — JSON in Markdown eingepackt, Schema nicht erzwungen:

```
content: [ {type: thinking, …}, {type: text, text: "```json\n{ \"title\": … }\n```"} ]
```

Dieselbe Anfrage über die **OpenAI-Route** derselben llama-swap-Instanz mit `response_format`:

```
content: '{"title": "npm-Anwendungen auf diesem PC anzeigen"}'
```

**Der Server kann strukturierte Ausgaben — nur nicht über das Feld, das die Anthropic-Route
schickt.** Da `claude-local` über `ANTHROPIC_BASE_URL` fest auf diese Route setzt, ist der
OpenAI-Weg nicht erreichbar. Ohne übersetzenden Proxy bleibt das so.

## Befund 5: Der Prompt-Cache funktioniert (kein Problem)

Zweimal dieselbe Anfrage mit 2418 Token stabilem Präfix:

| | Dauer |
|---|---:|
| Lauf 1 (kalt) | 3,20 s |
| Lauf 2 (identisches Präfix) | **0,64 s** |

**Faktor 5.** Der Server prefillt das gemeinsame Präfix einmal und rechnet danach nur die Differenz.

Daraus folgt: `CLAUDE_LOCAL_LEAN=1` (`--bare --exclude-dynamic-system-prompt-sections`) hilft
**nicht** durch weniger Tokens, sondern durch ein **stabileres Präfix**. Dynamische Abschnitte —
Zeitstempel, wechselnde Verzeichnisinhalte, MCP-Serverlisten — entwerten den Cache dahinter.

---

## Befund 6: Hintergrund-Requests liefen ins Leere (404)

Claude Code schickt neben der eigentlichen Konversation **Hintergrund-Anfragen** — Sitzungstitel,
kurze Klassifizierungen, Zusammenfassungen. Die gehen nicht an das gesetzte Modell, sondern an ein
zweites, das über `ANTHROPIC_SMALL_FAST_MODEL` bestimmt wird. Ohne diese Variable ist die Vorgabe
`claude-3-5-haiku-*` — ein Name, den die `config.yaml` nicht kennt.

Im llama-swap-Log:

```
11:34:58  404  POST /v1/messages   no model id could be identified
```

Das erklärt Hänger, die keinem Rechenaufwand entsprachen: die Anfrage wurde nie ausgeführt,
Claude Code wartete auf eine Antwort, die es nicht gab.

**Wichtig:** Dieser Fehler ist **stumm**. Er taucht nur im llama-swap-Zugriffslog auf, nicht in
der Claude-Code-Oberfläche und nicht im llama-server-Log — dort ist die Anfrage schlicht nicht
vorhanden, weil sie den Upstream nie erreicht. Wer nur den Modell-Log liest, sieht eine Lücke und
keine Ursache.

### Lösung: `ANTHROPIC_SMALL_FAST_MODEL` mitsetzen

In `~/.local/bin/claude-local`:

```bash
MODEL="${CLAUDE_LOCAL_MODEL:-claude}"
…
export ANTHROPIC_MODEL="$MODEL"
export ANTHROPIC_SMALL_FAST_MODEL="$MODEL"   # sonst 404 auf claude-3-5-haiku-*
```

Damit zeigen auch die Hintergrund-Anfragen auf einen gültigen Alias. Weil beide auf dasselbe
Modell zeigen, bleibt es bei einer Instanz — die `exclusive`-Gruppe wird nicht verletzt und es
gibt kein zusätzliches Laden.

Wer den Unterschied will, kann `ANTHROPIC_SMALL_FAST_MODEL` auf ein kleineres Modell zeigen lassen
(z. B. `qwen35-8b`). Dann lädt llama-swap allerdings ein zweites Modell, und in einer
`exclusive`-Gruppe verdrängt das das große — für diesen Aufbau also **nicht** empfehlenswert.

**Gegenprobe:** nach einem frischen `claude-local`-Aufruf im Log auf
`no model id could be identified` prüfen. Bleibt es aus, greift der Fix.

---

## Lösungen

### `--chat-template-kwargs '{"auto_disable_thinking_with_tools":true}'`

Das Chat-Template `qwen36-froggeric-v21.jinja` (28 KB) kennt diese Variable:

```jinja
{%- if auto_disable_thinking_with_tools and _has_tools %}
    {%- set ns_state.thinking = false %}
```

Sie schaltet das Nachdenken **bedingt** ab — nur wenn Werkzeuge mitgeschickt werden. Gemessen:

| | ohne | mit |
|---|---:|---:|
| Reasoning | 77 Zeichen | **0** |
| Tool-Call | 1 ✓ | 1 ✓ |
| Completion | 48 Tokens | **28** |
| Dauer | 1,67 s | **1,27 s** |

Gegenprobe ohne Werkzeuge: Reasoning 147 Zeichen, Antwort `OK` — der normale Chat behält seine
Denkphase. Das ist besser als ein reines Kappen, weil es situationsabhängig entscheidet.

### `--reasoning-budget 512` und `--no-reasoning-preserve`

Als Auffangnetz für den werkzeugfreien Fall und gegen das Kontextwachstum. Mit `max_tokens: 2048`
liefert dieselbe Frage, die vorher leer zurückkam:

| | Baseline (16000) | mit 512 |
|---|---|---|
| Antwort | **leer** | `ggml-sycl-rdna2.cpp` |
| Completion | 200, abgeschnitten | 522, regulär beendet |
| Dauer warm | — | 9,8 s |

### `max_tool_response_chars: 16000`

Das Template kürzt Werkzeug-Ergebnisse gutmütig: Anfang behalten, Hinweis anhängen, JSON-Nutzlasten
ausnehmen.

```jinja
{%- set content = content[:max_tool_response_chars]
    + '\n[TRUNCATED - original length ' ~ (content | length | string) ~ ' chars]' %}
```

Gemessen mit einem 32 900 Zeichen langen Ergebnis: **5 492 statt ~8 200 prompt_tokens.**

### Allowlist in `~/.claude/settings.json`

25 Regeln für lesende Routinebefehle ergänzt (auf 140 gesamt), damit der Classifier für sie gar
nicht erst befragt wird — unabhängig davon, ob das Modell gerade geladen ist.

**Bewusst nicht aufgenommen:** `ssh` (465 Aufrufe, häufigster überhaupt), `python3`, `docker exec`,
`uv tool` — allesamt arbiträre Codeausführung. Mutierende Befehle (`git add`, `rsync`,
`npm install`, `systemctl restart/start/stop`) ebenso; bei `systemctl` wurden nur die lesenden
Unterbefehle einzeln aufgenommen. `curl` nur mit Präfix auf lokale Hosts, sodass `-X POST` (steht
immer vor der URL) nicht greift. `rocm-smi` nur als `--show*`, weil das Werkzeug auch schreiben kann.

### Loopguard im Template

Im Skriptkopf von `claude-local` steht seit dem 2026-09-09 die Messung:

> *bei identischen Tool-Ergebnissen im Verlauf wiederholen ALLE getesteten Modelle ab n=5 denselben
> Aufruf endlos.*

Das Template hatte bereits einen Zähler für aufeinanderfolgende **Fehler** (`ns2.consecutive_failures`)
mit Warnungseinschub — aber keinen für identische **erfolgreiche** Ergebnisse.

Ergänzt in `qwen36-froggeric-v21-loopguard.jinja`:

```jinja
{%- set ns2 = namespace(prev_role='', consecutive_failures=0,
                        prev_tool_content='', repeat_count=0) %}
...
{%- if content == ns2.prev_tool_content %}
    {%- set ns2.repeat_count = ns2.repeat_count + 1 %}
{%- else %}
    {%- set ns2.repeat_count = 0 %}
{%- endif %}
{%- set ns2.prev_tool_content = content %}
...
{%- if ns2.repeat_count >= 2 %}
    {{- '⚠️ SYSTEM WARNING: identisches Werkzeug-Ergebnis zum N. Mal. …' }}
{%- endif %}
```

Verifiziert über die Template-Rendering-Route `/upstream/<model>/apply-template` mit drei
identischen Ergebnissen: **genau eine Warnung, beim dritten** — also vor dem gemessenen n=5.
Fehlerwarnungen wurden korrekt nicht ausgelöst. (Die Prüfung lief gegen einen inzwischen
entfallenen Testeintrag; dieselbe Templatedatei liegt heute in `claude`.)

---

## Endstand

`ornith` blieb auf dem ursprünglichen Stand — der Chat soll seine Denkphase behalten. Das gesamte
Agenten-Tuning liegt in einem **eigenen Eintrag `claude`**, der dasselbe Modell lädt:

```yaml
  ornith:                                  # unverändert, für Chat
    cmd: >
      ${hip_env_dual} ${server}
      --model ${models_dir}/Ornith-1.5-35B-A3B-FULLY-OBLITERATED.Q6_K.gguf
      --chat-template-file ${tmpl_dir}/qwen36-froggeric-v21.jinja
      --port ${PORT} -c 131072
      --tensor-split 5,1
      ${qwen_full}
    ttl: 3600

  claude:                                  # für Claude Code
    cmd: >
      ${hip_env_dual} ${server}
      --model ${models_dir}/Ornith-1.5-35B-A3B-FULLY-OBLITERATED.Q6_K.gguf
      --chat-template-file ${tmpl_dir}/qwen36-froggeric-v21-loopguard.jinja
      --port ${PORT} -c 131072
      --tensor-split 5,1
      ${qwen_full}
      --reasoning-budget 512
      --no-reasoning-preserve
      --chat-template-kwargs '{"auto_disable_thinking_with_tools":true,"max_tool_response_chars":16000}'
    ttl: 3600
    name: "Claude Code (Agent)"
```

Gemessen, identische Anfrage mit Werkzeugen:

| Modell | Reasoning | Tool-Call | Completion |
|---|---:|---:|---:|
| **`claude`** | **0** | 1 ✓ | 41 |
| `ornith` | 107 | 1 ✓ | 53 |

Der Loopguard liegt direkt in `claude` — er ist für den agentischen Betrieb gedacht, ein separater
Testeintrag ist damit überflüssig. Nutzung: `CLAUDE_LOCAL_MODEL=claude claude-local`, oder in
`claude-local` den Vorgabewert auf `claude` setzen.

Sicherungen der `config.yaml` liegen als `.bak` bis `.bak8` daneben; `llama-swap -validate`
bestätigt 8 Modelle.

---

### Profile über Aliase statt zweiter Modellinstanz

`llama-swap` kann Anfrageparameter **pro Alias** überschreiben (`filters.setParamsByID`). Die
Dokumentation nennt genau diesen Fall: *„Useful with aliases to vary behaviour depending on which
alias the client used (e.g. different reasoning_effort per alias)"*. Werte dürfen Objekte sein,
`chat_template_kwargs` lässt sich also injizieren.

```yaml
    aliases:
      - ornith-fast
      - ornith-think
    filters:
      setParamsByID:
        ornith-fast:
          chat_template_kwargs:
            reasoning_effort: "none"
            auto_disable_thinking_with_tools: true
            max_tool_response_chars: 16000
        ornith-think:
          chat_template_kwargs:
            auto_disable_thinking_with_tools: false
            reasoning_effort: "high"
            max_tool_response_chars: 16000
```

Gemessen, identische Anfrage mit Werkzeugen:

| Alias | Reasoning | Tool-Call | Completion |
|---|---:|---:|---:|
| `ornith` | 0 | 1 ✓ | 28 |
| `ornith-fast` | 0 | 1 ✓ | 28 |
| `ornith-think` | **64** | 1 ✓ | 47 |

**Alle drei laufen auf demselben geladenen Prozess** — der Wechsel kostet keine Ladezeit. Der
Unterschied zwischen `ornith` und `ornith-fast` zeigt sich erst *ohne* Werkzeuge: `ornith` denkt
dort weiter (147 Zeichen bei „Sage OK"), `ornith-fast` gar nicht.

> **Grenze:** Das Chat-Template wird über `--chat-template-file` beim **Serverstart** gewählt und
> ist kein Anfrageparameter. Aliase können es nicht umschalten — deshalb ist der Loopguard an den
> eigenen Eintrag `claude` gebunden und nicht per Alias von `ornith` erreichbar.
>
> Die hier gezeigten Aliase sind daher **nicht mehr konfiguriert**; die Trennung läuft über zwei
> Modelleinträge. Der Abschnitt bleibt als belegtes Verfahren stehen, falls du Profile ohne
> Template-Wechsel brauchst.

---

## Server-Flags durchgemessen (2026-09-11)

Vier Parameter der `config.yaml` einzeln vermessen. **Einer hat getragen, drei nicht.** Aufbau:
Produktionsbefehl des `claude`-Eintrags auf Port 18080 nachgebaut, `-sm layer` wie produktiv,
Prompt aus echten Quelldateien (33 988 Token) plus vier **verschiedene** Fragen, `temperature 0`,
`top_k 1`, je 600 Token Ausgabe.

### ✅ `--spec-draft-n-max` 2 → 3: **+19 % Decode**

Die einzige Verbesserung. Im Testaufbau +7,6 bis +15,6 % über drei vergleichbare Fragen, und im
**echten Betrieb** bestätigt:

| | n-max 2 | n-max 3 |
|---|---:|---:|
| `#mean acc len` | 1,90 | **2,59** |
| Annahmequote | 46,6 % | **52,9 %** |
| `#acc rate/pos` | 0,587 / 0,310 | 0,726 / 0,526 / **0,333** |
| Decode (Mittel) | ~40 t/s | **~47,6 t/s** |

Die dritte Draft-Position wird in **33 %** der Fälle angenommen — vorher wurde nach zwei Tokens
abgebrochen, auch wenn das dritte gepasst hätte. Pro Ausgabetoken etwa **30 % weniger
Verifikationsrunden**, also 30 % weniger vollständige Modell-Durchläufe.

`mean acc len = 1 + Σ(acc rate/pos)` — geprüft an beiden Messungen (1 + 0,587 + 0,310 = 1,897 ≈ 1,90;
1 + 0,726 + 0,526 + 0,333 = 2,585 ≈ 2,59). Diese Identität macht die Logzeile direkt auswertbar.

> ⚠ Die kumulative Statistik **steigt über die Sitzung**, das ist ein Artefakt. Die Einzelwerte je
> Anfrage schwanken zwischen 2,15 und 3,21 ohne Richtung; der kumulative Wert konvergiert nur, weil
> frühe kleine Stichproben an Gewicht verlieren. Nicht als „wird immer besser" lesen.

### ❌ `--spec-draft-n-max 4`: −13 %

| | n-max 3 | n-max 4 |
|---|---:|---:|
| `mean acc len` | 3,14 | 3,30 (+5 %) |
| erzeugte Draft-Tokens | 2287 | **2895 (+27 %)** |
| angenommene | 1632 | 1667 (+2 %) |
| Trefferquote | 71,4 % | **57,6 %** |
| Decode | 58,1 / 57,3 / 58,0 / 64,2 | **50,4 / 49,5 / 53,2 / 53,7** |

Position 4 wird durchaus in 33 % der Fälle angenommen, die Annahmelänge steigt sogar. **Ursache
des Verlusts: der Verifikationsbatch wächst von 4 auf 5 Positionen (+25 %)**, und bei 48k Kontext
mit Attention über den gesamten KV kostet das mehr als die 5 % längere Annahme einbringen.

### ❌ `--spec-draft-p-min > 0`: −4 bis −13 %

| `p-min` | `mean acc len` | Trefferquote | Decode kalt |
|---|---:|---:|---:|
| **0,00** | **3,14** | 71,4 % | **58,2** |
| 0,10 | 3,14 | 71,4 % | 57,6 |
| 0,50 | 3,04 | 80,2 % | 54,5 |
| 0,90 | 2,89 | **96,7 %** | 50,5 |

Bei 0,10 wird **nichts** gefiltert (Statistik bis zur letzten Stelle identisch) — der MTP-Kopf liegt
durchweg über 10 % Zuversicht. Ab 0,50 greift der Filter, und ab da wird es schlechter. Monoton,
kein Optimum dazwischen.

Die Trefferquote lässt sich auf 96,7 % treiben, **nur ist die Verschwendung fast gratis und die
Ersparnis teuer**: ein Draft-Token kostet ~2,6 ms, eine Verifikationsrunde ~46 ms. Wegfiltern
verkürzt die Annahmelänge und erzwingt mehr Runden (bei 0,90: 1034 statt 764 Draft-Aufrufe).

### ❌ `--ubatch-size 4096`: Folgeanfragen fast doppelt so lang

| | `-ub 2048` | `-ub 4096` |
|---|---:|---:|
| kalt, 33 988 Tok | **809,5 t/s** → 42,0 s | 783,5 t/s → 43,4 s |
| Folgeanfrage: zu rechnende Tokens | **2053** | **4097** |
| Folgeanfrage: Dauer | **3,58 s** | **6,92 s** |
| VRAM GPU 1 | — | **31,62 / 32,75 GB** |

Der Kaltstart verliert 3,2 % (Baseline über fünf Läufe 809,4–810,3, also klar außerhalb des
Rauschens). Der eigentliche Schaden ist aber die **Feinkörnigkeit der Cache-Wiederverwendung**: der
Prompt-Cache stellt nur bis zur Chunk-Grenze wieder her, und die Chunk-Größe ist die physische
Batchgröße. Ein größerer `-ub` verdoppelt damit die neu zu rechnenden Tokens bei Folgeanfragen —
genau dem Betriebspunkt, der im Agentenbetrieb zählt.

Dazu blieben nur **1,1 GB VRAM-Reserve** bei 34k Kontext; produktiv laufen 48k.

### Das Muster hinter den drei Fehlschlägen

**Entscheidend ist die Annahmelänge pro Verifikationsrunde.** Alles, was sie senkt, verliert —
verschwendetes Drafting ist nahezu gratis. Die beiden Richtungen scheitern spiegelbildlich:
`n-max 4` vergrößert den teuren Verifikationsbatch, `p-min` verkürzt die Annahme. `n-max 3`,
`p-min 0`, `-ub 2048` ist in allen drei Parametern das Optimum.

### Zwei Flags in der Config tun nichts

Aus dem Ladeprotokoll:

```
W srv load_model: cache_reuse is not supported by this context, it will be disabled
W cmn set_process_: failed to set process priority 2 : Permission denied (13)
```

`--cache-reuse 256` wird **stillschweigend abgeschaltet** — dieser Kontext unterstützt es nicht.
Die Präfix-Wiederverwendung läuft trotzdem, aber über die LCP-Slot-Auswahl
(`selected slot by LCP similarity, f_sim_best = 0.998`), nicht über dieses Flag. Und `--prio 2`
scheitert an fehlenden Rechten. Beides kann raus, sonst sucht man später an der falschen Stelle.

### `ngram-mod` liefert im Agentenbetrieb nichts — bleibt trotzdem drin

Drei Sitzungen, jeweils **0 angenommene Entwürfe** bei 150–290 Aufrufen. Es ist aber nicht kaputt,
sondern braucht **wörtlich wiederkehrende Passagen**: im Testaufbau, wo das Modell Quellcode aus
dem Prompt zitierte, erreichte es `#mean acc len = 28,5` bei Annahmeraten um 1,000 und trieb das
Decode auf 107 t/s.

Kosten laut Log: **36,7 ms gegen 11,5 s** für `draft-mtp` — ein Tausendstel. Da `mtp_args` ein
Makro für **alle** Modelle ist und im Chat mit Code-Zitaten nachweislich liefert, bleibt es aktiv.
Entfernen wäre Kosmetik.

> ⚠ **Testfehler, der zweimal zu ungültigen Messungen führte:** bei **identischen** Anfragen trifft
> `ngram-mod` seine eigene frühere Ausgabe (`mean acc len 28,5`, Annahmerate 1,000) und treibt tg
> auf 107 t/s. Auch mit verschiedenen Fragen stört es noch, wenn die Antwort Code aus dem Prompt
> zitiert. Für `n-max`- oder `p-min`-Vergleiche **`--spec-type draft-mtp` allein** fahren.

### Methodisches

- **Kontrollmessung ist Pflicht.** `n-max 3` wurde dreimal gefahren und lieferte **identische**
  Draft-Statistik (2287 / 1632 / 764 Runden, 3,14) bei tg innerhalb 0,5 %. Das Drafting ist bei
  `temperature 0` vollständig deterministisch — erst dadurch sind Differenzen von 3 % deutbar.
- **Vor dem Test `curl http://127.0.0.1:8080/unload`**, nicht `kill`. llama-swap gibt den VRAM
  sauber frei und läuft weiter. Danach das Modell mit einer Minimalanfrage wieder anwärmen, sonst
  zahlt der Nutzer den Kaltstart.
- `-sm` ist produktiv **nicht** gesetzt, also Layer-Split — passend zur `AGENTS.md`-Regel
  „MTP-Modelle → Layer-Split". Messungen mit `-sm tensor` sind nicht übertragbar.

---

## Was **nicht** gelöst ist

**Die Schleifenbildung selbst.** Der Loopguard rendert nachweislich die Warnung — ob sie ein
35B-A3B aus dem Muster holt, ist offen und nur im echten Betrieb feststellbar. Nutzung über
`CLAUDE_LOCAL_MODEL=claude claude-local`.

**Strukturierte Ausgaben.** `output_config` fehlt serverseitig. Betrifft die Sitzungsbenennung
(kosmetisch) und alles, was Anthropic künftig darüber löst. Nicht konfigurierbar. Hinweis: dass
die Sitzungsbenennung nicht funktionierte, hatte **zwei** Ursachen — diese hier und das 404 aus
Befund 6. Nur letzteres ließ sich beheben.

**Der Kaltstart — und er ist der größte verbleibende Posten.** Gemessen am 2026-09-11:

```
prompt eval time = 66499.35 ms / 48187 tokens (724.62 t/s)
```

**66,5 Sekunden**, mit fallendem Durchsatz von 1258 auf 737 t/s über die Kontextlänge. Eine Woche
vorher waren es 38 463 Token und 52,6 s — der Prompt wächst.

Zum Vergleich: der gesamte MTP-Gewinn (`n-max 3`) spart etwa **eine halbe Sekunde** pro
Folgeanfrage. Der Kaltstart liegt zwei Größenordnungen darüber.

> ⚠ **Korrektur einer früheren Annahme:** Ich hatte diese großen Prefills dem System-Prompt
> zugeschrieben und daraus `CLAUDE_LOCAL_LEAN=1` als Hauptlösung abgeleitet. Das Log widerlegt das
> teilweise — vor dem 48k-Aufruf lief ein Aufruf mit nur 739 Token. Die 48k sind also **angehäufte
> Konversation samt gelesener Dateien**, nicht nur der System-Prompt. LEAN
> (`--bare --exclude-dynamic-system-prompt-sections`) kürzt nur den System-Prompt-Anteil und kann
> den Kaltstart daher **nicht beseitigen**, nur verkleinern. Wie viel, ist **ungemessen**.

`ttl: 0` würde das Nachladen vermeiden, ist aber durch die `exclusive`-Gruppe begrenzt — für zwei
27-GB-Modelle nebeneinander reicht der VRAM nicht.

**Modellqualität.** Ein 35B-A3B ist für agentische Werkzeugketten schwächer als die Modelle, für
die Claude Code gebaut wurde. Keine Konfiguration ändert das.

---

## Fallstricke fürs nächste Mal

**Das Denk-Budget bindet nur, wenn `max_tokens` größer ist.** Bei knappem Limit frisst das
Nachdenken die ganze Antwort und man bekommt einen leeren String zurück — reproduziert bei
`max_tokens: 60`. Claude Code setzt 32000, andere Werkzeuge womöglich nicht.

**llama.cpp-eigene Endpunkte gehen über `/upstream/<model>/…`.** Ein direkter Aufruf von
`/apply-template` antwortet mit 404 — daraus wurde hier zunächst fälschlich geschlossen, llama-swap
reiche solche Routen gar nicht durch. Es gibt den dokumentierten Pfad
`/upstream/:model_id/…`; verifiziert mit `/upstream/ornith/props` → HTTP 200. Der Umweg über den
Upstream-Port aus der Prozessliste war unnötig.

**Logs liegen unter `/logs`, nicht `/logs/upstream`.** Auch hier führte der falsche Pfad zu einem
404 und zum voreiligen Schluss, die Logs seien unerreichbar. `/logs` liefert die gepufferten
Zugriffslogs, `/logs/stream/upstream` streamt die Upstream-Prozesse. **Offen bleibt:** In `/logs`
fanden sich keine llama.cpp-Timings (`prompt eval time`, `t/s`) — bei `logLevel: info` gibt
llama-server sie womöglich gar nicht aus.

**Erst die Dokumentation, dann der Schluss.** Zwei der ursprünglich sechs vermeintlichen Lücken in
llama-swap existierten nicht; sie entstanden daraus, dass ein 404 als „Funktion fehlt" gelesen
wurde statt als „falscher Pfad".

**Quoting in `config.yaml` überlebt.** `--chat-template-kwargs '{"…":true}'` in einem
YAML-`>`-Block kommt korrekt beim Server an — llama-swap entfernt die einfachen Anführungszeichen.
Geprüft über `ps -eo args`.

**Vor jeder Messung VRAM prüfen.** Ein Benchmark gegen belegten Speicher scheitert mit
`GGML_ASSERT(meta_buf_ctx->bufs[i])` — einem verkleideten OOM, das wie eine Regression aussieht.

**Claude Code benutzt zwei Modellnamen, nicht einen.** `ANTHROPIC_MODEL` allein reicht nicht;
`ANTHROPIC_SMALL_FAST_MODEL` fällt sonst auf `claude-3-5-haiku-*` zurück und erzeugt 404er (Befund
6). Bei unerklärlichen Hängern **zuerst das llama-swap-Zugriffslog** auf
`no model id could be identified` durchsehen — nicht das llama-server-Log, dort fehlt die Anfrage
vollständig. Faustregel: eine Wartezeit ohne passenden Eintrag im Modell-Log ist kein
Rechenproblem, sondern ein Routing-Problem.
