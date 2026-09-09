# Claude Code gegen den lokalen Server: Diagnose und Behebung

**Untersucht:** 2026-09-09 · **Maschine:** x99 (2× MI50, gfx906) · **Stack:** llama-swap → mx-org-Build

Claude Code lief gegen den lokalen `ornith`-Endpunkt spürbar zäh: lange Wartezeiten, abgebrochene
Werkzeugaufrufe, blockierte Bash-Befehle. Die Ursache war **nicht** die Hardware — Prefill und
Decode liegen auf dieser Maschine bei ~1000 bzw. ~55 t/s. Es waren fünf voneinander unabhängige
Probleme in der Konfiguration und im Protokollpfad.

---

## Aufbau

`~/.local/bin/claude-local` setzt `ANTHROPIC_BASE_URL=http://192.168.178.71:8080` und ruft
`claude --model ornith` auf. Claude Code spricht damit die **Anthropic-Messages-API** (`/v1/messages`),
die llama.cpp emuliert. Dahinter steht llama-swap mit sieben Modellen in einer `exclusive`-Gruppe.

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

Verifiziert über `/apply-template` am Upstream-Server (Port 11004, llama-swap reicht diese Route
nicht durch) mit drei identischen Ergebnissen: **genau eine Warnung, beim dritten** — also vor dem
gemessenen n=5. Fehlerwarnungen wurden korrekt nicht ausgelöst.

---

## Endstand

```yaml
  ornith:
    cmd: >
      ${hip_env_dual} ${server}
      --model ${models_dir}/Ornith-1.5-35B-A3B-FULLY-OBLITERATED.Q6_K.gguf
      --chat-template-file ${tmpl_dir}/qwen36-froggeric-v21.jinja
      --port ${PORT} -c 131072
      --tensor-split 5,1
      ${qwen_full}
      --reasoning-budget 512
      --no-reasoning-preserve
      --chat-template-kwargs '{"auto_disable_thinking_with_tools":true,"max_tool_response_chars":16000}'
```

Dazu `ornith-loopguard` als Testeintrag mit dem gepatchten Template, `ttl: 600`, `-c 32768`.
Sicherungen der `config.yaml` liegen als `.bak` bis `.bak6` daneben.

---

## Was **nicht** gelöst ist

**Die Schleifenbildung selbst.** Der Loopguard rendert nachweislich die Warnung — ob sie ein
35B-A3B aus dem Muster holt, ist offen und nur im echten Betrieb feststellbar. Nutzung über
`CLAUDE_LOCAL_MODEL=ornith-loopguard claude-local`.

**Strukturierte Ausgaben.** `output_config` fehlt serverseitig. Betrifft die Sitzungsbenennung
(kosmetisch) und alles, was Anthropic künftig darüber löst. Nicht konfigurierbar.

**Der Kaltstart.** Nach `ttl: 3600` kostet der erste Aufruf ~60 s. `ttl: 0` würde helfen, ist aber
durch die `exclusive`-Gruppe begrenzt — für zwei 27-GB-Modelle nebeneinander reicht der VRAM nicht.

**Modellqualität.** Ein 35B-A3B ist für agentische Werkzeugketten schwächer als die Modelle, für
die Claude Code gebaut wurde. Keine Konfiguration ändert das.

---

## Fallstricke fürs nächste Mal

**Das Denk-Budget bindet nur, wenn `max_tokens` größer ist.** Bei knappem Limit frisst das
Nachdenken die ganze Antwort und man bekommt einen leeren String zurück — reproduziert bei
`max_tokens: 60`. Claude Code setzt 32000, andere Werkzeuge womöglich nicht.

**`llama-swap` reicht nur die Modell-Routen durch.** `/apply-template` und andere llama.cpp-eigene
Endpunkte antworten mit 404; dafür den Upstream-Port aus der Prozess-Kommandozeile nehmen
(`ps -eo args | grep -o -- "--port [0-9]*"`).

**Quoting in `config.yaml` überlebt.** `--chat-template-kwargs '{"…":true}'` in einem
YAML-`>`-Block kommt korrekt beim Server an — llama-swap entfernt die einfachen Anführungszeichen.
Geprüft über `ps -eo args`.

**Vor jeder Messung VRAM prüfen.** Ein Benchmark gegen belegten Speicher scheitert mit
`GGML_ASSERT(meta_buf_ctx->bufs[i])` — einem verkleideten OOM, das wie eine Regression aussieht.
