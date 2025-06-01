# Instagram Videos Downloader

A simple website/API for downloading Instagram videos, built with Next.js. It works seamlessly and allows you to download videos with no hassle.


## Description

This website allows you to easily download Instagram videos in MP4 format. Simply paste the URL of any public Instagram post, and you'll receive the video file. There's also an API that you can integrate into your own applications to download Instagram videos programmatically. The API returns JSON responses with the video URL and other metadata.

_Note: Instagram Stories aren't supported._

You can preview and try the website live on Vercel here: [instagram-reels-downloader-tau.vercel.app](https://instagram-reels-downloader-tau.vercel.app/)

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/riad-azz/Instagram-reels-downloader.git
```

### 2. Install dependencies

```bash
cd Instagram-reels-downloader
npm install
```

### 3. Start the server

For development:

```bash
npm run dev
```

For production (build and start):

```bash
npm run build
npm run start
```

### Endpoint: `/api/video?postUrl={POST_URL}`

Parameters:

- `postUrl` : Instagram post or reel link **(required)**.

#### GET Request example

```bash
curl -i "http://localhost:3000/api/video?postUrl=https://www.instagram.com/reel/DCUBzY0yiKK/"
```

#### API Response

```json
{
  "status": "success",
  "data": {
    "filename": "ig-downloader-1712666263.mp4",
    "width": "640",
    "height": "640",
    "videoUrl": "https://scontent.cdninstagram.com/o1/v/t16/f1/m84/E84E5DFC48EA8...etc"
  }
}
```

## Rate Limiter - Upstash

To optimize API performance and reduce the load, rate limiting has been implemented using Upstash. This limits the number of requests to the API within a specific time frame to avoid service disruptions.

To enable rate limiting, follow these steps:

1. Create an account on [upstash.com](https://upstash.com/).
2. Create a new Redis database.
3. Click on the newly created database.
4. Under "REST API", click on `.env` and copy the provided variables.
5. Create a `.env.local` file in the root directory.
6. Paste the variables into the `.env.local` file and add the following line:
   ```env
   USE_UPSTASH="true"
   UPSTASH_REDIS_REST_URL="YOUR-UPSTASH-URL"
   UPSTASH_REDIS_REST_TOKEN="YOUR-UPSTASH-TOKEN"
   ```

All rate-limit configurations can be found in `src/features/ratelimit/constants.ts`.

If you want to change the identifier (default is IP), you can modify it in `src/middleware.ts`.

## License

This project is licensed under the **Apache License 2.0**. See the LICENSE.md file for details.
