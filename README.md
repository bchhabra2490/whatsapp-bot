# WhatsApp Receipt Capture Bot

A WhatsApp bot that captures photos and PDFs (receipts, business cards, notes), runs OCR, stores records and embeddings in Supabase, and answers questions using semantic search and an OpenAI-powered agent.

## Features

- **Media capture** — Send images or PDFs via WhatsApp; they are stored in Supabase Storage and processed with Mistral OCR.
- **Embeddings** — OCR text and text notes are embedded (OpenAI) and stored in Postgres (pgvector) for semantic search.
- **Q&A agent** — Ask questions in natural language; the bot uses tools to search your records and answers via OpenAI.
- **Conversation context** — Recent messages are stored and used for intent detection and richer answers.
- **Background processing** — Incoming messages are handled asynchronously with Celery and Redis; the webhook responds immediately and replies when processing finishes.

## Tech Stack

| Layer        | Technology                    |
|-------------|-------------------------------|
| Backend     | Python 3.8+, Flask            |
| WhatsApp    | Twilio WhatsApp Webhooks     |
| OCR         | Mistral API (Pixtral)        |
| Embeddings  | OpenAI text-embedding-3-small|
| Chat/Agent  | OpenAI API (e.g. gpt-4o-mini)|
| Database    | Supabase Postgres + pgvector |
| Storage     | Supabase Storage             |
| Queue       | Celery + Redis               |

## Prerequisites

- **Python** 3.8+ (3.11+ recommended)
- **Redis** (for Celery broker/backend)
- **Twilio** account with WhatsApp (sandbox or business)
- **Supabase** project (Postgres + Storage)
- **Mistral** API key
- **OpenAI** API key
- **Poppler** (for PDFs): `brew install poppler` (macOS) or `apt-get install poppler-utils` (Linux)

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-org/whatsapp-bot.git
cd whatsapp-bot

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment variables

Copy the example below into a `.env` file in the project root. Do not commit `.env`.

```env
# Twilio (WhatsApp)
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886

# Supabase
SUPABASE_URL=
SUPABASE_KEY=
SUPABASE_STORAGE_BUCKET=whatsapp
SUPABASE_RECORDS_TABLE=wbot_records
SUPABASE_MESSAGES_TABLE=wbot_messages
SUPABASE_JOBS_TABLE=wbot_jobs

# Mistral OCR
MISTRAL_API_KEY=
MISTRAL_MODEL=pixtral-12b-2409

# OpenAI
OPENAI_API_KEY=
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Celery / Redis
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# App
FLASK_ENV=production
FLASK_DEBUG=0
PORT=5000
```

### 3. Supabase setup

1. In Supabase: **Database → Extensions** → enable **vector**.
2. In **SQL Editor**, run the full script in `database/schema.sql` (creates `wbot_records`, `wbot_messages`, `wbot_jobs`, and the `wbot_match_records` RPC).
3. In **Storage**, create a bucket named `whatsapp` (or set `SUPABASE_STORAGE_BUCKET` accordingly) and set policies so your app can upload.

### 4. Twilio setup

1. In Twilio: **Messaging → Try it out → Send a WhatsApp message** (or use a WhatsApp Sender).
2. Note the WhatsApp “From” number (e.g. `+1 415 523 8886`) and set `TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886`.
3. Set your webhook URL to `https://your-domain.com/webhook` (must be HTTPS).

### 5. Run the app

Start Redis, then the web app and the Celery worker (from the project root):

```bash
# Terminal 1: Redis (if not running as a service)
redis-server

# Terminal 2: Flask
python app.py

# Terminal 3: Celery worker (loads .env from project root)
celery -A services.tasks.celery_app worker --loglevel=info
```

For production, run Flask with gunicorn and Celery with the appropriate concurrency and broker settings.

## How it works

1. **Webhook** — Twilio sends incoming WhatsApp messages (text or media) to `POST /webhook`.
2. **Job creation** — The handler writes the incoming message to `wbot_messages`, creates a row in `wbot_jobs` with status `queued`, and enqueues a Celery task. It replies immediately with a short “processing” style message.
3. **Background task** — The Celery worker loads the job, sets status to `processing`, runs either media handling (download → storage → OCR → embed → save to `wbot_records`) or text handling (intent → save note or answer question via the agent). It then updates the job to `completed` or `failed` and sends the final WhatsApp reply via the Twilio API.

## Usage

- **Media** — Send one or more images or PDFs; the bot stores them, runs OCR, embeds the text, and replies when done.
- **Text** — Send a message to save a note (intent: save) or to ask a question (intent: question). The agent can call tools to search your records and uses recent conversation history for context.

## Project structure

```
whatsapp-bot/
├── app.py                    # Flask app, webhook, job enqueue
├── services/
│   ├── tasks.py              # Celery app and process_whatsapp_job task
│   ├── supabase_client.py    # Supabase DB, storage, jobs, messages
│   ├── mistral_ocr.py        # Mistral OCR
│   ├── openai_client.py      # OpenAI embeddings + chat/agent
│   ├── receipt_processor.py  # Media/note pipeline + agent tools
│   └── whatsapp_handler.py   # handle_media / handle_text
├── database/
│   └── schema.sql            # Tables and wbot_match_records RPC
├── requirements.txt
├── setup_venv.sh
├── verify_setup.py
└── README.md
```

## API

- `GET /health` — Health check.
- `POST /webhook` — Twilio WhatsApp webhook (accepts Twilio form-encoded body).

## Data model (summary)

- **wbot_records** — Stored items (media or note): `phone_number`, `record_type`, `storage_urls` / `ocr_text` / `user_text`, `embedding`, metadata.
- **wbot_messages** — Conversation history: `phone_number`, `direction`, `role`, `content`, `message_sid`.
- **wbot_jobs** — Async jobs: `phone_number`, `message_sid`, `job_type` (media/text), `payload`, `status` (queued → processing → completed/failed), `result` / `error`.

## Troubleshooting

- **Celery “SUPABASE_URL/SUPABASE_KEY not set”** — Run the worker from the project root so `services/tasks.py` can find `.env` (it calls `load_dotenv()` in `make_celery()`).
- **Twilio 21910 / “Invalid From and To pair”** — Use WhatsApp for both sides: `TWILIO_WHATSAPP_NUMBER=whatsapp:+1...` and ensure the “To” number is in `whatsapp:+...` form when sending.
- **Twilio “could not find a Channel with the specified From address”** — Use the exact WhatsApp “From” number from your Twilio WhatsApp sandbox or Sender (e.g. `whatsapp:+14155238886`).
- **Python 3.14 and httpcore** — Use `httpcore>=1.0.8` (and matching httpx) if you see `AttributeError` related to `typing.Union` / `__module__`.

## License

MIT
