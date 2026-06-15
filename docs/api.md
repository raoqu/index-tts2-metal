# MTTS HTTP API

This document describes the native `--server` HTTP API.

## Start Server

```bash
./build/mtts --server \
  --host 127.0.0.1 \
  --port 3456 \
  --model_bundle bin \
  --voice_store voices \
  --queue_size 16 \
  --tts_concurrency 1 \
  --clone_concurrency 1
```

Start with the embedded web management UI:

```bash
./build/mtts --web \
  --webkey change-me \
  --host 127.0.0.1 \
  --port 3456 \
  --model_bundle bin \
  --voice_store voices
```

`--web` implies `--server` and exposes `GET /web`. If only `--server` is specified, `/web` is not exposed.

Options:

- `--host`: bind address. Default: `127.0.0.1`.
- `--port`: bind port. Default: `3456`.
- `--model_bundle`: MIT2 model bundle directory. Default: `MODEL_BUNDLE` env or `bin`.
- `--voice_store`: directory for sqlite metadata, uploaded samples, and generated voice bundles. Default: `MIT2_VOICE_STORE` env or `voices`.
- `--queue_size`: max waiting requests per TTS/clone queue. Default: `MIT2_QUEUE_SIZE` env or `16`.
- `--tts_concurrency`: configured TTS worker concurrency. Default: `MIT2_TTS_CONCURRENCY` env or `1`.
- `--clone_concurrency`: configured clone worker concurrency. Default: `MIT2_CLONE_CONCURRENCY` env or `1`.
- `--web`: enable the embedded web management UI at `/web`.
- `--webkey`: login key for `/web` and `/web/api/*`. If omitted, `MIT2_WEBKEY` is used when set.

Note: the current server executes TTS/clone work serially for Metal runtime stability. Queue and concurrency options are accepted as the API contract, but effective synthesis is single-filed.

## Health

### `GET /health`

```bash
curl http://127.0.0.1:3456/health
```

Response:

```json
{"status":"ok"}
```

`GET /v1/health` is also supported.

## Web Management UI

### `GET /web`

Available only when the process is started with `--web`.

```bash
curl http://127.0.0.1:3456/web
```

The page is embedded in the `mtts` binary. It provides:

- Voice list, import, clone-from-audio, edit, and delete.
- Speech test using a voice ID or direct voice path.
- Runtime status, current job, queue length, totals, and recent jobs.

The browser UI stores the entered web key in local storage and sends it as `X-MTTS-Web-Key` to `/web/api/*`.

### Web Login Check

```bash
curl -X POST http://127.0.0.1:3456/web/api/login \
  -H "Content-Type: application/json" \
  -d '{"key":"change-me"}'
```

Response:

```json
{"ok":true}
```

### Web Runtime Status

```bash
curl http://127.0.0.1:3456/web/api/status \
  -H "X-MTTS-Web-Key: change-me"
```

Response:

```json
{
  "status": "ok",
  "queue": {
    "max_waiting": 16,
    "waiting": 0,
    "running": true,
    "current": {
      "id": 1,
      "kind": "tts",
      "label": "你好世界",
      "elapsed_seconds": 1.01323
    }
  },
  "totals": {
    "submitted": 1,
    "completed": 0,
    "failed": 0,
    "rejected": 0
  },
  "recent": []
}
```

`GET /api/status` is also available without web authentication for API clients.

## Voice Management

Voice records are stored in sqlite under `--voice_store`. Audio samples are stored under `<voice_store>/samples`, and generated native MIT2 voice bundles are stored under `<voice_store>/bundles`.

### List Voices

`GET /api/voices`

```bash
curl http://127.0.0.1:3456/api/voices
```

Response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "voice_...",
      "object": "audio.voice",
      "name": "qin",
      "description": "local voice",
      "bundle_path": "sample/qin.pt",
      "sample_path": "",
      "source_audio_seconds": 0,
      "source": "import",
      "created_at": "1781277150",
      "updated_at": "1781277150"
    }
  ]
}
```

`GET /v1/audio/voices` is also supported.

### Preview Original Source Audio

`GET /api/voices/{voice_id}/source-audio`

Returns `audio/wav` containing the original reference audio embedded in the native `.pt` voice bundle. For imported legacy records that do not have embedded preview audio, the server falls back to `sample_path` when available.

```bash
curl http://127.0.0.1:3456/api/voices/voice_xxx/source-audio \
  --output source-preview.wav
```

### Import Existing Voice Bundle

`POST /api/voices`

```bash
curl -X POST http://127.0.0.1:3456/api/voices \
  -H "Content-Type: application/json" \
  -d '{
    "name": "qin",
    "description": "local voice",
    "bundle_path": "sample/qin.pt"
  }'
```

Response:

```json
{
  "id": "voice_...",
  "object": "audio.voice",
  "name": "qin",
  "description": "local voice",
  "bundle_path": "sample/qin.pt",
  "sample_path": "",
  "source_audio_seconds": 0,
  "source": "import",
  "created_at": "1781277150",
  "updated_at": "1781277150"
}
```

### Create Voice From Reference Audio

`POST /api/voices`

```bash
curl -X POST http://127.0.0.1:3456/api/voices \
  -F name=demo-voice \
  -F description="created from reference audio" \
  -F audio_sample=@sample/qin.wav
```

Accepted multipart file field names: `audio_sample`, `recording`, `file`, or `audio`.

`POST /v1/audio/voices` supports the same JSON and multipart forms.

### Get Voice

`GET /api/voices/{voice_id}`

```bash
curl http://127.0.0.1:3456/api/voices/voice_xxx
```

### Update Voice

`PATCH /api/voices/{voice_id}`

```bash
curl -X PATCH http://127.0.0.1:3456/api/voices/voice_xxx \
  -H "Content-Type: application/json" \
  -d '{
    "name": "new-name",
    "description": "updated description"
  }'
```

Updatable fields:

- `name`
- `description`
- `bundle_path`
- `sample_path`

### Delete Voice

`DELETE /api/voices/{voice_id}`

```bash
curl -X DELETE http://127.0.0.1:3456/api/voices/voice_xxx
```

Response:

```json
{"deleted":true}
```

## OpenAI-Compatible Speech

### `POST /v1/audio/speech`

Generates WAV audio from text.

Request fields:

- `model`: accepted for OpenAI compatibility. The server uses `--model_bundle`.
- `input`: required text.
- `voice`: optional. May be a string voice ID/path or an object with `id`.
- `response_format`: optional. Only `wav` is supported.
- `output`: optional server-side output WAV path. If omitted, the response body is `audio/wav`.
- `stream`: optional boolean. If `output` is set, default response is JSON ack. Set `stream: true` to also return WAV bytes.

Use a voice ID:

```bash
curl -X POST http://127.0.0.1:3456/v1/audio/speech \
  -H "Content-Type: application/json" \
  -o output.wav \
  -d '{
    "model": "mtts",
    "input": "你好世界",
    "voice": {"id": "voice_xxx"},
    "response_format": "wav"
  }'
```

Use a direct voice bundle path:

```bash
curl -X POST http://127.0.0.1:3456/v1/audio/speech \
  -H "Content-Type: application/json" \
  -o output.wav \
  -d '{
    "model": "mtts",
    "input": "你好世界",
    "voice": "sample/qin.pt",
    "response_format": "wav"
  }'
```

Save to a server-side output file and return JSON:

```bash
curl -X POST http://127.0.0.1:3456/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mtts",
    "input": "你好世界",
    "voice": {"id": "voice_xxx"},
    "response_format": "wav",
    "output": "/tmp/mtts_output.wav"
  }'
```

Response:

```json
{
  "status": "ok",
  "output": "/tmp/mtts_output.wav",
  "audio_seconds": 1.75311,
  "total_seconds": 15.4829,
  "rtf": 8.83167
}
```

Save to a server-side file and also return audio bytes:

```bash
curl -X POST http://127.0.0.1:3456/v1/audio/speech \
  -H "Content-Type: application/json" \
  -o output.wav \
  -d '{
    "model": "mtts",
    "input": "你好世界",
    "voice": {"id": "voice_xxx"},
    "response_format": "wav",
    "output": "/tmp/mtts_output.wav",
    "stream": true
  }'
```

## OpenAI-Compatible Voice Consents

### Create Consent And Voice

`POST /v1/audio/voice_consents`

Uploads a consent/reference recording. By default, the server also creates a local voice and returns `voice_id`.

```bash
curl -X POST http://127.0.0.1:3456/v1/audio/voice_consents \
  -F name=demo-voice \
  -F language=zh \
  -F recording=@sample/qin.wav
```

Response:

```json
{
  "id": "consent_...",
  "object": "audio.voice_consent",
  "name": "demo-voice",
  "language": "zh",
  "recording_path": "voices/samples/sample_....wav",
  "voice_id": "voice_...",
  "created_at": "1781277321",
  "updated_at": "1781277321"
}
```

Accepted multipart file field names: `recording`, `audio_sample`, `file`, or `audio`.

### Create Consent Only

```bash
curl -X POST http://127.0.0.1:3456/v1/audio/voice_consents \
  -F name=demo-voice \
  -F language=zh \
  -F create_voice=false \
  -F recording=@sample/qin.wav
```

### List Consents

```bash
curl http://127.0.0.1:3456/v1/audio/voice_consents
```

### Get Consent

```bash
curl http://127.0.0.1:3456/v1/audio/voice_consents/consent_xxx
```

### Update Consent

```bash
curl -X PATCH http://127.0.0.1:3456/v1/audio/voice_consents/consent_xxx \
  -H "Content-Type: application/json" \
  -d '{
    "name": "new-name",
    "language": "zh"
  }'
```

### Delete Consent

```bash
curl -X DELETE http://127.0.0.1:3456/v1/audio/voice_consents/consent_xxx
```

Response:

```json
{"deleted":true}
```

## Errors

Errors are JSON:

```json
{
  "error": {
    "message": "voice not found or not a usable voice bundle"
  }
}
```

Common status codes:

- `400`: missing or unsupported request field.
- `404`: unknown endpoint or record not found.
- `429`: TTS or clone queue is full.
- `500`: synthesis, clone, sqlite, or filesystem error.
