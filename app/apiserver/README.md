# audiocpp_apiserver

A headless HTTP API for music generation and editing. No UI, no embedded
assets, no browser routes — a web front end is one client among several, and it
lives somewhere else.

Every request that touches a model becomes a **job**: submitted, queued, run on
a worker, and written to disk. Submission returns immediately with an id; the
work happens afterwards and its record outlives the process.

## Why this exists alongside `audiocpp_server`

`audiocpp_server` is synchronous. `handle_generic_run` blocks the connection
thread for the whole generation — a minute or more for a song — and its only
concurrency control is a per-model mutex that returns 503 when busy. Nothing can
be cancelled; `busy_guard.h` says so in its own comment. And nothing is
remembered: when the process ends, so does every record of what it produced.

That server is untouched and still builds. This one answers three different
questions:

| | `audiocpp_server` | `audiocpp_apiserver` |
|---|---|---|
| Long request | Holds the connection open | Returns an id in milliseconds |
| Busy | 503 | Queued, with a position |
| Cancel | Not possible | Reaches the model's inner loop |
| Crash | Everything is lost | Job, request and outputs survive |
| Client knowledge | Hardcoded per family | Read from `/v1/capabilities` |

## Build

```powershell
pwsh scripts\build_windows.ps1 -Preset windows-cuda-release -Target audiocpp_apiserver
```

The binary lands in `build\windows-cuda-release\bin\audiocpp_apiserver.exe`.

## Config

Takes the same file `audiocpp_server` does, so an existing config works
unchanged. Extra settings live under an `api` object, which the old server
ignores.

```jsonc
{
  "host": "127.0.0.1",
  "port": 8781,
  "backend": "cuda",
  "device": 0,
  "threads": 8,
  "lazy_load": true,
  "max_loaded_models": 1,
  "min_free_memory_mb": 512,

  "models": [
    {
      "id": "yue2-3b-q8",
      "family": "yue2",
      "task": "gen",
      "mode": "offline",
      "path": "C:/cmodels/yui",
      "model_spec_override": "C:/src/audiocpp-api/model_specs/yue2.json",
      "lazy": true,
      "session_options": {
        "yue2.model_gguf": "yue2-3b-q8_0.gguf",
        "yue2.vae_gguf": "yue2-vae-f32.gguf"
      }
    }
  ],

  "api": {
    "workers": 1,
    "data_dir": "C:/cmodels/yui/api-data",
    "max_queued": 64,
    "retention_days": 30,
    "unload_after_cancel": true
  }
}
```

| `api` key | Default | Meaning |
|---|---:|---|
| `workers` | `1` | Concurrent jobs. See the note below before raising it. |
| `data_dir` | `<config dir>/.audiocpp-api` | Where jobs and outputs live |
| `max_queued` | `256` | Submissions past this get 503 |
| `retention_days` | `30` | `0` disables age-based collection |
| `max_jobs_retained` | `5000` | `0` disables |
| `max_bytes_retained` | 50 GiB | `0` disables |
| `requeue_on_start` | `true` | Re-queue jobs that were waiting when the server died |
| `requeue_max_age_hours` | `24` | Older waiting jobs are marked interrupted instead |
| `unload_after_cancel` | `true` | Drop the session after a cancel rather than reusing it |
| `unload_after_error` | `true` | Drop it after a failed run too. A run that throws unwinds past the session's own cleanup, so its graph arenas stay resident and the next job fails on a smaller allocation |
| `unload_after_job` | `false` | Drop it after every job. Costs one reload; worth it where peak VRAM sits close to the card |
| `oom_retries` | `2` | Times a job that ran out of device memory is retried on a halved memory budget. `0` disables |
| `progress_interval_ms` | `250` | Throttle for progress events |
| `max_event_subscribers` | `64` | Each SSE stream costs a connection thread |

### Running out of device memory

Every yue2 stage allocates its whole KV cache up front, for the window it was
asked for: one f32 K and one f32 V per layer per token, 224 KiB a token on the
3B. The stock windows -- 4096 planner tokens, 9000 semantic, a 24576 acoustic
window -- are sized for a card several times larger than a 6 GB one, so a run
would decode for minutes and then die on `failed to allocate ... graph`. The
same request fails the same way on upstream `audiocpp_server`; it is not a leak
and not specific to this server.

The server handles this in two layers, and neither needs configuring:

1. **Each stage is measured against the device before it starts.** Free memory
   is read immediately before the planner, the semantic decode and the acoustic
   solve, and that stage's window is clamped to what fits. Measuring per stage
   rather than once means each answer already accounts for what the earlier
   stages still hold. A card with room clamps nothing.
2. **A run that runs out anyway is retried on a smaller budget.** It stays the
   same job; the model is dropped, the budget halves, and it starts again.
   `oom_retries` (default 2) bounds this, so the worst case is a budget
   quartered rather than an endless loop. Each retry is logged.

The budget is split between the prefix a run must carry and the tokens it will
generate, because they buy different things. A prefix token is prefilled and
held: it pays a cache, a share of a prefill graph sized to the whole prefix,
and a prefix-state at the acoustic stage that keeps it as f32, bf16 and f16 at
once. A generated token pays its cache and a share of the NAR graph arena, the
largest single allocation in a run. Measured on the 3B, that is five caches per
prefix token and six per generated token; the run's own peaks are logged per
stage (`[yue2] semantic peak ... 2.95 caches`) so the weights can be rechecked
against any card rather than trusted.

This split matters most under `cot=full`, where the planner writes a score of
hundreds to thousands of tokens that becomes the prefix. Charged at the
generated rate, that plan alone exhausts a budget that holds it easily at the
prefix rate, and the take collapses to the 200-token floor -- eight seconds --
while the same request at `cot=off` runs for minutes.

The cache is f16 on CUDA, HIP and Vulkan and f32 on CPU; `yue2_ar_cache_bytes_per_element`
is the one place that decides it, and the budget reads it rather than assuming.

Windows that were shortened appear in the job's `result` output under `notes`,
so a short take on a small card says why rather than looking like the model
stopped early.

Three request options override the estimate when you know better:

| Option | Default | Effect |
|---|---:|---|
| `nar_context` | `24576` | Acoustic window in tokens; the chunk solved at once is `(nar_context - prefix - 3) / 2` |
| `vram_headroom_mb` | `100` | Held back from the estimate for graph arenas and allocator slack. Never more than half of what is free. Low by default, so the score and the take run as long as the card allows; raise it when a run's notes report spilling to system RAM |
| `vram_budget_scale` | `1.0` | Multiplies every window the budget allows. The lever the retry escalates |
| `acoustic_single_window` | `false` | Solve the take as one acoustic window rather than letting the budget split it. Removes the seam a chunk boundary leaves; a take too large for the card then spills to system RAM and the stage runs several times slower, which the run's notes say. An out-of-memory retry gives this up first, before it touches `vram_budget_scale` |

Measured on an RTX 4050 (6 GB), everything at stock defaults: prompts that had
failed render 2:02, 3:23 and 2:37 of audio, peaking at 5.2-5.5 GiB. The cost is
length -- on a card this size the budget lands well short of the model's
six-minute ceiling.

`--host`, `--port`, `--workers`, `--data-dir` and `--cors-origins` override the
file.

### About `workers`

The default of 1 is not a placeholder. Two jobs for the same model **always**
serialise, whatever this is set to, because a model has one session and that
session holds mutable graph arenas and KV state. Extra workers only help across
*different* models.

On a 6 GB card running yue2 there is no room for a second resident generation at
all — the weights alone are over 3 GB before arenas. The server refuses to start
when `workers` exceeds `max_loaded_models`, because that configuration makes a
worker evict a model another worker is running, which fails that job instead of
waiting. A clear error at startup beats intermittent 503s under load.

## Start

```powershell
.\build\windows-cuda-release\bin\audiocpp_apiserver.exe --config C:\cmodels\yui\api.json
```

```
[api] store C:/cmodels/yui/api-data: 7 jobs loaded, 1 marked interrupted
[api] restored 1 queued jobs
[api] listening on http://127.0.0.1:8781 with 1 worker
```

That first line is the point of the whole thing: the previous process died
mid-run, and its jobs are still here.

## The job lifecycle

```
POST /v1/jobs ──► queued ──► running ──► succeeded
                    │           │    └──► failed
                    │           └───────► cancelled
                    └──────────────────► cancelled   (cancelled before it started)

                        server dies ───► interrupted
```

`interrupted` is deliberately separate from `failed`. It means the server stopped
while the job was running — an operational fact, not a model fault, and the only
terminal state for which offering "resubmit" makes sense.

## Endpoints

Full schemas in [`docs/openapi.yaml`](../../docs/openapi.yaml), also served from
`GET /openapi.yaml`.

| Method | Path | |
|---|---|---|
| `GET` | `/health` | Liveness, queue depth, residency |
| `GET` | `/v1/capabilities` | What each model can actually do |
| `GET` | `/v1/models` | Configured models and whether they are loaded |
| `POST` | `/v1/models/load` · `/unload` | Explicit residency control |
| `POST` | `/v1/jobs` | Submit; returns 202 with an id |
| `POST` | `/v1/tasks` | Same, for a family-native request passed through unchanged |
| `GET` | `/v1/jobs` | History, filtered and paged |
| `GET` | `/v1/jobs/{id}` | One job, with live progress |
| `GET` | `/v1/jobs/{id}/result` | Outputs once terminal |
| `GET` | `/v1/jobs/{id}/outputs/{name}` | Download one output |
| `GET` | `/v1/jobs/{id}/events` | SSE: progress, stages, completion |
| `POST` | `/v1/jobs/{id}/cancel` | Stop it |
| `PATCH` | `/v1/jobs/{id}` | Pin it, or edit its metadata |
| `DELETE` | `/v1/jobs/{id}` | Remove it and its files |
| `GET` | `/v1/queue` | What is running and what is waiting |
| `POST` | `/v1/uploads` | Stage one file for a job to use as input |

### Paths never travel in a request

Every filename a client sends has to name a file inside the job's own input
directory: not absolute, no drive letter, no `..`. The CLI's rule is the
opposite -- it hands an absolute path straight through -- and that is right for a
CLI and an arbitrary file read over HTTP, on a server that binds a loopback port
with `cors_origins: "*"`. Send a file first and use the name that comes back:

```bash
curl -s -X POST "http://127.0.0.1:8781/v1/uploads?name=take.wav"      -H 'Content-Type: audio/wav' --data-binary @take.wav
# {"id":"up_01M2HPS2Z25P7B04HYR7FEE6S5","name":"take.wav","bytes":9576748}

curl -s -X POST http://127.0.0.1:8781/v1/jobs -H 'Content-Type: application/json' -d '{
  "model": "sheetsage2", "operation": "transcribe",
  "uploads": ["up_01M2HPS2Z25P7B04HYR7FEE6S5"],
  "request": {"audio": "take.wav"}
}'
```

The worker copies the upload into `<job>/input/` before the request is built, so
what a family opens is always a plain name in one directory the server owns.
`api.max_upload_mb` caps a single file, and defaults to 128.

Errors keep the shape the existing server uses:

```json
{"error": {"message": "job 'job_01M2…' is already succeeded", "type": "conflict"}}
```

`400 invalid_request_error`, `404 not_found`, `409 conflict`, `503 server_busy` /
`insufficient_memory`, `500 server_error`. The 409 is new here: it means the
request was well-formed but the target was in the wrong state, which is not
something a client can fix by changing the request.

## Walkthrough: generate, then edit the music

This is the workflow the YuE2 model card calls agentic editing. It was not
reachable before, because the score the model composes was generated, used as an
internal prefix, and thrown away.

**1. Generate.**

```bash
curl -s -X POST http://127.0.0.1:8781/v1/jobs \
  -H 'Content-Type: application/json' -d '{
    "model": "yue2-3b-q8",
    "operation": "generate",
    "metadata": {"title": "take 1"},
    "request": {
      "text": "[verse]\nneon rain across the causeway\nthe city hums a lower key\n",
      "options": {
        "style": "dreamy synth pop, female vocal, 96 bpm",
        "cot": "full",
        "seed": "831001",
        "semantic_max_tokens": "700"
      }
    }
  }'
```

```json
{"id":"job_01M2CFWZ8QMX3T7Y1H5S6EYPR1","status":"queued","queue_position":0, ...}
```

**2. Watch it.**

```bash
curl -N http://127.0.0.1:8781/v1/jobs/job_01M2CFWZ8QMX3T7Y1H5S6EYPR1/events
```

```
event: progress
id: 4
data: {"type":"progress","stage":"score","step":48,"total":4096,"fraction":0.0117}
```

Stages run `score → semantic → acoustic → decode`. The semantic stage is most of
the wall time.

**3. Read the score the model wrote.**

```bash
curl -s .../outputs/score
```

```abc
X:1
M:4/4
L:1/16
Q:1/4=97
V: Vocal clef=treble name="Vocal Melody"
V: Ins   clef=treble name="Ins Melody"
K:Dm
% verse
V: Vocal
"Dm"z8d2e4f2-|"Dm"f4e2d6c2c2-|"Bbmaj7"c2d4z8z2|
V: Ins
Z2|z8z2d4c2-|c2d4c4d4c2|
```

**4. Change it and re-render.** Transposing to A minor, keeping the lyrics and
style:

```bash
sed -e 's/^K:Dm/K:Am/' -e 's/"Dm"/"Am"/g' -e 's/"Bbmaj7"/"Fmaj7"/g' score.abc > edited.abc
```

Submit it back with `cot: "melody"`, which tells the model to honour the score
rather than plan a new one:

```json
{
  "model": "yue2-3b-q8",
  "operation": "score_render",
  "request": {
    "text": "[verse]\nneon rain across the causeway\nthe city hums a lower key\n",
    "options": {
      "style": "dreamy synth pop, female vocal, 96 bpm",
      "cot": "melody",
      "abc": "<the edited score>"
    }
  }
}
```

Same words, same voice, different harmony. It is also considerably faster — 23 s
against 165 s in one measured pair — because supplying a score skips the
planning stage entirely. That last point has a consequence worth
knowing: on a run that supplies `abc`, every `abc_*` sampling option is inert,
because the planner they configure never executes.

**4b. Or use the score as an idea rather than an arrangement.** `abc_mode`
decides what a supplied score is for:

| `abc_mode` | What happens |
|---|---|
| `strict` (default) | The ABC block is closed after your score. The planner never runs and the model sings exactly what you gave it |
| `seed` | The block is left open. Your score becomes the opening of the planner's own context, and the model carries on writing it -- same key, same meter, same feel, because it is reading its own prefix -- then sings the whole thing |

So `seed` turns eight bars into a full song built on them. The score that comes
back in `outputs/score.abc` is the whole of what was sung, your bars included,
so it can be edited and re-rendered like any other.

```json
{"options": {"style": "bossa nova, nylon guitar", "cot": "full",
             "abc": "<eight bars>", "abc_mode": "seed"}}
```

**5. Replay it.** A finished job also emits `semantic_codes`, the tokens behind
its audio. Supplying them through `semantic_codes_file` **replaces** generation
with those tokens and re-renders them -- useful for re-decoding a take without
singing it again, and for teacher forcing while debugging.

It does not extend anything. `generate_semantic` copies the codes and returns
before the model runs, so nothing is continued. Real continuation means feeding
the codes to the AR as a prefix after `kMusicStartToken` instead of as a
replacement, and that is not implemented. To make a take longer today, give the
singing a larger window -- lower `vram_headroom_mb`, or a shorter score sharing
the budget -- and read the job notes, which now say when a take stopped because
it hit its ceiling rather than because it ended.

## What each family can do

Read this from `/v1/capabilities` rather than assuming — the families genuinely
differ, and the API does not paper over it.

| Family | Audio input | Editing |
|---|---|---|
| `yue2` | **No** | Through the symbolic score: read the ABC, change it, re-render |
| `sheetsage2` | Yes, and only | Reads a recording and writes it as a score; composes nothing |
| `ace_step` | Yes | `cover`, `repaint`, `extract`, `lego`, `complete` — real region repaint |
| `stable_audio` | Yes | init-audio and multi-region inpaint |

yue2 has no audio-conditioning path at all: `Yue2Request` has no audio field,
`vae2llm`/`llm2vae` are internal NAR projections rather than an audio→AR route,
and there is no semantic tokenizer to turn audio into codes. So a yue2 "cover"
means supplying an ABC score with a new style, and region repaint that rejoins
the surrounding audio is not achievable for that family. It is achievable with
`ace_step`, which is why the capability is reported per model.

`sheetsage2` is the other half of that answer. It turns a recording into an ABC
score -- melody, chords, beats, sections -- in the same dialect yue2 reads and
writes, which is what makes a cover a round trip rather than a conversion:
transcribe the recording, edit the score, then hand it back to yue2 with
`cot=melody` and a style of your own. It reads audio and composes nothing, so it
is `task: "midi"` and never appears as something to generate with.

It wants its own directory. The family finds its weights by looking for the one
GGUF where the model lives, so a folder holding several of them is ambiguous and
refused by name:

```json
{ "id": "sheetsage2", "family": "sheetsage2", "task": "midi",
  "path": "C:/cmodels/yui/SheetSage2-GGUF", "lazy": true }
```

## Cancellation, honestly

`POST /v1/jobs/{id}/cancel` returns:

- **200** if the job was still queued — decided there and then, already terminal.
- **202** if it was running — the cancel has been delivered; the terminal status
  follows.

ggml has no CUDA abort callback, so a graph already executing cannot be
interrupted. Cancellation is polled *between* work units instead, which means it
lands at the next boundary rather than instantly:

| Stage | Unit | Practical latency |
|---|---|---|
| `score`, `semantic` | one decoded token | tens of ms — **measured 184 ms** end to end, mid-semantic |
| `acoustic` | one flow-matching step over the whole sequence | up to seconds |
| `decode` | one decoder tile (~40 s of audio) | up to seconds |

Since the semantic stage dominates a generation, the common case is fast. The
API does not promise a flat number, because the code cannot deliver one.

After a cancel the model is unloaded by default (`unload_after_cancel`). Unwinding
out of a generation loop skips the cleanup the success path does, leaving a
partially advanced KV cache and arenas resident; a reload costs about ten
seconds and removes the doubt.

## The store

```
<data_dir>/
  version                       refuse to start on an unknown one
  jobs/<id>/job.json            status, progress, outputs, timings
  jobs/<id>/request.json        the client's body, written once, never rewritten
  jobs/<id>/events.ndjson       append-only event log, replayed by SSE
  jobs/<id>/outputs/audio.wav
  jobs/<id>/outputs/score.abc
  jobs/<id>/outputs/semantic_codes.bin
  jobs/.corrupt/<id>/           quarantine
  jobs/.trash/<id>/             rename-then-delete staging
```

Ids are ULIDs, so a directory listing is already chronological and history paging
is a plain cursor comparison with no offset scan.

`job.json` is written to a temp file and renamed over the target, so a torn
record is not possible. `request.json` is deliberate redundancy with the
`request` field inside it: if the record is ever damaged, the lyrics are still
there.

**On startup** the store reconciles what the last process left behind. A job
marked `running` under a different run id becomes `interrupted`. A job still
`queued` is re-queued, unless it is older than `requeue_max_age_hours` — nobody
wants a week-old queue replaying after a reboot. An unparseable record is moved
to `.corrupt/` rather than stopping the server, because one bad job must not
turn a durability feature into an availability bug.

**Retention** runs at startup and on an interval: oldest terminal jobs first,
until every configured bound is satisfied. Pinned, queued and running jobs are
never collected.

## Tests

391 cases across eleven files, each source carrying its own direct-execution
entry point behind a file-specific macro:

```powershell
pwsh scripts\build_windows.ps1 -Preset windows-cuda-release -RunTests -ConfigureOnly
cmake --build build\windows-cuda-release -j 16 --target apiselftest_job apiselftest_job_store ...
ctest --test-dir build\windows-cuda-release -R "^apiselftest_" --output-on-failure
```

```
100% tests passed, 0 tests failed out of 11
```

Each binary also runs standalone and prints its inputs, outputs and expectations
per case, so a failure shows what it was given rather than only that it failed.

What they cover: job ids and their validation (including traversal attempts),
the on-disk round trip, crash reconciliation, retention, queue ordering and the
model-claim rule, cancellation latency against a fake that polls every 5 ms,
progress throttling, SSE framing and ring eviction, route matching, the error
mapping, and the eviction and concurrency-config decisions.

The end-to-end behaviour that needs a real model — a generation, the score
round-trip, a mid-run cancel, and the restart — is verified against the yue2
model at `C:\cmodels\yui`.

## Notes for whoever builds the UI

- Build the interface from `/v1/capabilities`, not from hardcoded family names.
  Adding a family should not require a client change.
- Subscribe to `/v1/jobs/{id}/events` rather than polling. Send `Last-Event-ID`
  on reconnect and nothing is missed.
- Never inline audio. The result gives you descriptors with URLs; a
  four-minute track is around 46 MB, and 61 MB once base64'd.
- The UI is a separate origin now, so a dev server has to be listed in
  `cors_origins` explicitly.
