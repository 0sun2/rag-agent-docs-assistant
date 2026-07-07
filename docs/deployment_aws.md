# AWS 배포 가이드 (Phase A: Lift & Shift)

로컬 Docker Compose 데모를 **실제 URL로 접속 가능한 AWS 서비스**로 올리는 절차.
핵심 설계 결정: GPU 의존 컴포넌트(BGE 임베딩 1.3GB + 리랭커 2.3GB)를 **Bedrock 관리형으로 스왑**해
GPU 인스턴스(g4dn ≈ 월 $380+) 대신 t3.large(≈ 월 $60)로 운영한다.

| 컴포넌트 | 로컬 (유지) | AWS 버전 (스왑) |
|---|---|---|
| LLM | OpenAI / Bedrock 선택 | Bedrock Claude (`LLM_PROVIDER=bedrock`) |
| 임베딩 | BGE 로컬 GPU | Titan Embeddings V2 (`PROD_EMBEDDING_PROVIDER=bedrock`) |
| 리랭커 | bge-reranker (GPU) | 생략 (`PROD_USE_RERANKER=false`) — hybrid만으로 hit@10 0.88~0.90 |
| 벡터스토어 | Chroma | v1: EC2 위 Chroma 유지 → v2: OpenSearch Serverless |

## 0. 사전 준비 (로컬 또는 EC2)

임베딩 모델이 바뀌면 **전체 재색인 필수** (색인·검색 임베딩 동일 원칙):

```bash
# Titan V2로 recursive 전략 재색인 (호출당 과금, 1,522문서 기준 수 달러 수준)
uv run python -m src.rag.embedding.run \
  --provider bedrock --model amazon.titan-embed-text-v2:0 --strategies recursive
```

재색인 후 기존 평가 파이프라인(hit@k/MRR, RAGAS)을 Titan 구성으로 재실행해
성능 하락 폭을 `experiments/`에 문서화한다 — 하락 자체가 "관리형 vs 셀프호스팅
임베딩 품질-비용 트레이드오프" 실험 섹션이 된다.

## 1. 인프라 (콘솔/CLI, ap-southeast-1)

1. **VPC**: 퍼블릭 서브넷 1개 + IGW. 보안그룹 인바운드 80/443만 (SSH는 내 IP 한정).
2. **EC2 t3.large** (2 vCPU / 8GB — 리랭커 없으면 충분) + **EIP**.
3. **IAM 역할** 부여 (인스턴스 프로파일): `bedrock:InvokeModel`, `bedrock:InvokeModelWithResponseStream`, `bedrock:Converse`, `bedrock:ConverseStream` 최소 권한.
   **자격증명 키를 인스턴스에 두지 않는다** — boto3가 인스턴스 메타데이터에서 역할을 해석한다.

## 2. 애플리케이션 기동

```bash
git clone <repo> && cd llm-docs-assistant
cp .env.example .env   # OPENAI/TAVILY 키 등 채우기 (LLM은 IAM 역할이라 키 불필요)

# AWS 오버라이드로 기동 (Bedrock LLM + Titan 임베딩 + rerank 생략 + CPU)
docker compose -f docker-compose.yml -f docker-compose.aws.yml up -d --build
```

`docker-compose.aws.yml`이 하는 일: `LLM_PROVIDER=bedrock`, `PROD_EMBEDDING_PROVIDER=bedrock`,
`PROD_USE_RERANKER=false`, GPU 예약 제거, `restart: unless-stopped`.

## 3. Nginx 리버스 프록시 + HTTPS

```nginx
server {
    listen 443 ssl;
    server_name demo.example.com;
    # certbot이 채우는 ssl_certificate ...

    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        # SSE 스트리밍(/agent/thread/stream) 필수 설정
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
    location / {
        proxy_pass http://127.0.0.1:8501/;
        proxy_http_version 1.1;              # Streamlit websocket
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Let's Encrypt: `sudo certbot --nginx -d demo.example.com`

## 4. 완료 기준 체크리스트

- [ ] 외부 URL에서 RAG QA + 에이전트 채팅(스트리밍 포함) 동작
- [ ] EC2에 자격증명 파일 없음 (`aws sts get-caller-identity`가 역할로 응답)
- [ ] 재부팅 후 자동 기동 (`restart: unless-stopped`)
- [ ] Budgets 알람 $20/50/100 3단계 설정 (과금 사고 방지)

## Phase B 이후 (요약)

- **대화 메모리 → DynamoDB**: SqliteSaver → DynamoDB checkpointer (`thread_id` 파티션 키, 온디맨드 ≈ $0)
- **Bedrock Guardrail**: PROMPT_ATTACK + PII 정책 생성 후 `.env`에 `BEDROCK_GUARDRAIL_ID` 설정
  → 입력 선검사 자동 활성 (`src/agent/security/input_guard.py`, 이미 구현됨)
- **벡터스토어 → OpenSearch Serverless**: BM25+kNN 하이브리드 네이티브. 단 유휴 OCU 고정비(월 $100+)
  주의 — 데모 트래픽이면 Chroma 유지가 더 싸다. 실험 후 컬렉션 삭제.

예상 비용(데모, 월): EC2 $60 + EBS/EIP $5~10 + Bedrock <$5 ≈ **~$70**.
안 쓸 때 인스턴스 중지로 절감.
