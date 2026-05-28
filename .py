import httpx


result = httpx.post("http://localhost:3001/read", json={
    "query": "produção de biodiesel",
    "top_k": 5,
    "min_score": 0.0
})
for r in result.json()["results"]:
    print(f"{r['score']:.4f} — {r['text']}")
    