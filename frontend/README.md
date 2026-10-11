# Content Moderation Tester

A small React + Vite + Tailwind UI for sending text and images to the existing FastAPI moderation backend.

## Run locally

### Docker Compose

From the project root, copy `.env.example` to `.env`, set the required database secret values, and start the stack:

```bash
docker compose up -d --build
```

Open the tester at <http://localhost:3000>. The frontend container serves the built React app through Nginx and embeds `http://localhost:8000/api/v1` as the browser API URL. The API remains available at <http://localhost:8000/docs>.

Stop the stack with:

```bash
docker compose down
```

### Without Docker

Start the backend first from the project root:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:3000>.

The frontend uses `http://localhost:8000/api/v1` by default. To use a different backend URL, copy `.env.example` to `.env.local`, change `VITE_API_BASE_URL`, and restart Vite:

```bash
cp .env.example .env.local
```

## Backend endpoints used

- `POST /api/v1/moderate/text` with JSON `{ "text": "..." }`
- `POST /api/v1/moderate/image` with multipart field `image`

The UI renders the backend's `prediction` labels and scores plus its `routing.action` / status decision. It does not add authentication, persistence, or any new backend routes.

## Production build preview

```bash
npm run build
npm run preview
```
