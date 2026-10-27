from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
import hashlib
import io
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from sqlalchemy import JSON, DateTime, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
import jwt
from docx import Document
from retrieval import embed_text, hybrid_search
from redis_backend import CacheBackend
from oidc import OIDCVerifier

BASE_DIR = Path(__file__).resolve().parent
LAWS_DB_PATH = BASE_DIR / "laws_db.json"
OPENAI_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
OPENAI_MODEL_NAME = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(BASE_DIR / 'legal_agent.db').as_posix()}")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-only-change-me")
JWT_ALGORITHM = "HS256"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
ALLOWED_ORIGINS = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",") if x.strip()]
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("legal-agent")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
REQUEST_COUNTER = Counter("legal_agent_requests_total", "Total API requests", ["path", "status"])
REQUEST_LATENCY = Histogram("legal_agent_request_latency_seconds", "API latency", ["path"])
IDEMPOTENCY_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
RATE_BUCKET: dict[str, list[float]] = {}
CACHE = CacheBackend()
OIDC = OIDCVerifier()


class Base(DeclarativeBase):
    pass


class CaseRecord(Base):
    __tablename__ = "cases"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    title: Mapped[str] = mapped_column(String(200), default="租赁纠纷案件")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(String(64), index=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32))
    output_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class ReviewRecord(Base):
    __tablename__ = "review_records"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    analysis_id: Mapped[str] = mapped_column(String(64), index=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    reviewer: Mapped[str] = mapped_column(String(128))
    decision: Mapped[str] = mapped_column(String(32))
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


Base.metadata.create_all(engine)

_ID_CARD_PATTERN = re.compile(r"(?<!\d)(\d{17}[0-9Xx]|\d{15})(?!\d)")
_PHONE_PATTERN = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")


def redact_pii(text: str) -> str:
    """在日志和模型上下文前做最小化脱敏，保留字段可读性。"""
    text = _ID_CARD_PATTERN.sub("[身份证号已脱敏]", text)
    return _PHONE_PATTERN.sub("[手机号已脱敏]", text)

app = FastAPI(title="AI 法律案件辅助分析平台 API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-Request-ID"])


class AnalyzeRequest(BaseModel):
    case_description: str = Field(..., min_length=10, max_length=20_000)
    claims: List[str] = Field(..., min_length=1, max_length=10)

    @field_validator("claims")
    @classmethod
    def validate_claims(cls, values: List[str]) -> List[str]:
        cleaned = [value.strip() for value in values if value and value.strip()]
        if not cleaned:
            raise ValueError("至少需要一个有效诉求")
        if any(len(value) > 500 for value in cleaned):
            raise ValueError("单条诉求不能超过 500 字")
        return cleaned


class TokenRequest(BaseModel):
    tenant_id: str = Field(default="demo-tenant", min_length=1, max_length=128)
    user_id: str = Field(default="demo-user", min_length=1, max_length=128)


class ReviewRequest(BaseModel):
    decision: Literal["APPROVED", "CHANGES_REQUESTED", "REJECTED"]
    comment: str = Field(default="", max_length=2_000)


def issue_token(tenant_id: str, user_id: str) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode({"sub": user_id, "tenant_id": tenant_id, "exp": now + timedelta(hours=8)}, JWT_SECRET, algorithm=JWT_ALGORITHM)


def current_identity(authorization: str | None = Header(default=None)) -> dict[str, str]:
    if not authorization:
        if os.getenv("REQUIRE_AUTH", "false").lower() == "true":
            raise HTTPException(status_code=401, detail="需要 Bearer Token")
        return {"user_id": "anonymous", "tenant_id": "demo-tenant"}
    try:
        scheme, token = authorization.split(" ", 1)
        if scheme.lower() != "bearer":
            raise ValueError
        if OIDC.enabled():
            payload = OIDC.verify(token)
            return {"user_id": str(payload.get("sub", "unknown")), "tenant_id": str(payload.get("tenant_id", payload.get("org_id", "default")))}
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {"user_id": str(payload["sub"]), "tenant_id": str(payload["tenant_id"])}
    except Exception as error:
        raise HTTPException(status_code=401, detail="无效的认证令牌") from error


def enforce_rate_limit(identity: dict[str, str]) -> None:
    if CACHE.url and not CACHE.allow(f"rate:{identity['tenant_id']}", RATE_LIMIT_PER_MINUTE):
        raise HTTPException(status_code=429, detail="请求频率超过当前租户配额")
    now = time.time()
    key = identity["tenant_id"]
    values = [stamp for stamp in RATE_BUCKET.get(key, []) if stamp > now - 60]
    if len(values) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="请求频率超过当前租户配额")
    values.append(now)
    RATE_BUCKET[key] = values


class Citation(BaseModel):
    evidence_id: str
    type: Literal["law", "case"]
    title: str
    article: str = ""
    quote: str
    source_version: str
    retrieval_score: float = Field(ge=0, le=1)
    synthetic: bool = False


class Issue(BaseModel):
    issue_id: str
    question: str
    position: str
    confidence: float = Field(ge=0, le=1)
    evidence_ids: List[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    analysis_id: str = ""
    request_id: str
    status: Literal["SUCCESS", "DEGRADED", "NEEDS_MORE_INFO", "MODEL_OUTPUT_INVALID"]
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]
    disclaimer: str
    laws: List[str]
    issues: Dict[str, List[str]]
    structured_issues: List[Issue] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    next_actions: List[str] = Field(default_factory=list)
    draft: str
    corpus_version: str
    model: str


def load_laws_db() -> Dict[str, Any]:
    if not LAWS_DB_PATH.exists():
        raise FileNotFoundError(f"未找到法律资料库文件: {LAWS_DB_PATH}")
    with LAWS_DB_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)
    metadata = data.get("metadata", {})
    if not metadata.get("version") or not metadata.get("source_url"):
        raise ValueError("法律资料库缺少版本或来源信息")
    for item in data.get("laws", []):
        if not item.get("source_url") or not item.get("effective_from"):
            raise ValueError(f"法律资料 {item.get('id')} 缺少来源或生效日期")
    return data


def _tokens(text: str) -> set[str]:
    text = text.lower()
    tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9_]+", text))
    for i in range(len(text) - 1):
        if "\u4e00" <= text[i] <= "\u9fff" and "\u4e00" <= text[i + 1] <= "\u9fff":
            tokens.add(text[i : i + 2])
    return tokens


def retrieve_relevant_context(case_description: str, claims: List[str], db: Dict[str, Any]) -> Dict[str, Any]:
    """可解释的轻量混合检索：关键词/二元词命中 + 文档类型加权。"""
    query_tokens = _tokens(f"{case_description} {' '.join(claims)}")

    def rank(items: List[Dict[str, Any]], fields: List[str], boost: float) -> List[tuple[float, Dict[str, Any]]]:
        ranked: List[tuple[float, Dict[str, Any]]] = []
        for item in items:
            searchable = " ".join(str(item.get(field, "")) if not isinstance(item.get(field, ""), list) else " ".join(map(str, item.get(field, []))) for field in fields)
            overlap = len(query_tokens & _tokens(searchable))
            score = min(1.0, overlap / max(1, min(10, len(query_tokens))) * boost)
            ranked.append((score, item))
        return sorted(ranked, key=lambda pair: pair[0], reverse=True)

    laws = [(item.rerank_score, item.document) for item in hybrid_search(f"{case_description} {' '.join(claims)}", db.get("laws", []), top_k=len(db.get("laws", [])))]
    cases = [(item.rerank_score, item.document) for item in hybrid_search(f"{case_description} {' '.join(claims)}", db.get("cases", []), top_k=len(db.get("cases", [])))]
    if not laws:
        laws = rank(db.get("laws", []), ["title", "content", "relevance"], 1.0)
    if not cases:
        cases = rank(db.get("cases", []), ["summary", "holding", "keywords"], 0.9)
    top_laws = [item for score, item in laws if score > 0][:5] or db.get("laws", [])[:3]
    top_cases = [item for score, item in cases if score > 0][:3] or db.get("cases", [])[:1]
    return {"laws": top_laws, "cases": top_cases, "scores": {"laws": laws, "cases": cases}}


def build_citations(context: Dict[str, Any], corpus_version: str) -> List[Citation]:
    citations: List[Citation] = []
    def bounded_score(value: Any) -> float:
        return round(min(1.0, max(0.0, float(value))), 4)
    score_by_id = {item.get("id"): score for score, item in context.get("scores", {}).get("laws", [])}
    score_by_id.update({item.get("id"): score for score, item in context.get("scores", {}).get("cases", [])})
    for item in context.get("laws", []):
        citations.append(Citation(evidence_id=f"ev_{item.get('id', uuid.uuid4().hex[:8])}", type="law", title=f"{item.get('source', '')}{item.get('article', '')}{item.get('title', '')}", article=item.get("article", ""), quote=item.get("content", ""), source_version=corpus_version, retrieval_score=bounded_score(score_by_id.get(item.get("id"), 0.1)), synthetic=bool(item.get("synthetic", False))))
    for item in context.get("cases", []):
        citations.append(Citation(evidence_id=f"ev_{item.get('id', uuid.uuid4().hex[:8])}", type="case", title=f"{item.get('case_title', '')}（{item.get('case_number', '')}）", quote=item.get("holding", item.get("summary", "")), source_version=corpus_version, retrieval_score=bounded_score(score_by_id.get(item.get("id"), 0.1)), synthetic=bool(item.get("synthetic", True))))
    return citations


def extract_json_text(raw_text: str) -> str:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip())
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            json.loads(match.group(0))
            return match.group(0)
    raise ValueError("模型返回内容不是合法 JSON")


def get_openai_client() -> OpenAI:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY 未配置")
    return OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL, timeout=LLM_TIMEOUT_SECONDS, max_retries=0)


def build_default_draft(claims: List[str]) -> str:
    return f"一、诉讼请求：请求法院支持以下诉求：{'；'.join(claims)}。二、事实与理由：现有材料显示双方存在房屋租赁关系，争议集中在退租、押金结算及房屋状态认定。三、举证重点：租赁合同、押金及租金支付记录、入住和退房交接材料、聊天记录、照片视频及维修票据。四、风险提示：提前退租责任、损失实际发生及正常损耗边界仍需结合原始证据核验。"


def build_analysis(request_id: str, payload: AnalyzeRequest, context: Dict[str, Any], *, status: str, model_result: Dict[str, Any] | None = None) -> AnalyzeResponse:
    metadata = load_laws_db().get("metadata", {})
    corpus_version = metadata.get("version", "unknown")
    citations = build_citations(context, corpus_version)
    law_refs = [citation.title for citation in citations]
    favorable = ["出租人无法证明具体损失、维修必要性或金额时，承租人可主张返还相应押金", "轻微污渍、自然磨损等正常使用后的状态通常不当然构成赔偿责任"]
    unfavorable = ["若合同约定提前退租责任，承租人可能承担约定违约金或可证明的实际损失", "若有交接照片、维修票据等证据证明异常损坏，押金可能被合理抵扣"]
    draft = build_default_draft(payload.claims)
    structured = [Issue(issue_id="issue_001", question="提前退租是否构成违约", position="需要结合合同中的提前退租条款和实际损失判断", confidence=0.78, evidence_ids=[c.evidence_id for c in citations[:2]]), Issue(issue_id="issue_002", question="押金扣款是否有依据", position="应核验扣款项目、损失证据和正常损耗边界", confidence=0.84, evidence_ids=[c.evidence_id for c in citations[:3]])]
    if model_result:
        draft = model_result.get("draft") if isinstance(model_result.get("draft"), str) else draft
        raw_issues = model_result.get("issues", {})
        if isinstance(raw_issues, dict):
            favorable_values = raw_issues.get("favorable", favorable)
            unfavorable_values = raw_issues.get("unfavorable", unfavorable)
            if isinstance(favorable_values, list):
                favorable = [str(x) for x in favorable_values if str(x).strip()]
            if isinstance(unfavorable_values, list):
                unfavorable = [str(x) for x in unfavorable_values if str(x).strip()]
    if len(payload.case_description) < 40:
        status = "NEEDS_MORE_INFO"
    return AnalyzeResponse(request_id=request_id, status=status, risk_level="MEDIUM", disclaimer="本结果仅供辅助分析，不能替代律师意见；正式使用前请由专业人士审核。", laws=law_refs, issues={"favorable": favorable, "unfavorable": unfavorable}, structured_issues=structured, citations=citations, missing_evidence=["书面租赁合同", "提前退租违约条款", "押金金额和支付记录", "出租人的具体扣款理由", "入住时房屋照片", "退租时房屋照片", "维修清单、发票或检测报告"], next_actions=["上传租赁合同并标注押金及违约金条款", "补充房屋交接和维修证据"], draft=draft, corpus_version=corpus_version, model=OPENAI_MODEL_NAME if model_result else "local-fallback")


def call_llm_analysis(payload: AnalyzeRequest, context: Dict[str, Any]) -> Dict[str, Any]:
    client = get_openai_client()
    prompt = {"case_description": redact_pii(payload.case_description), "claims": [redact_pii(x) for x in payload.claims], "retrieved_evidence": [{"id": x.get("id"), "content": x.get("content", x.get("holding", ""))} for x in context.get("laws", []) + context.get("cases", [])]}
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            completion = client.chat.completions.create(model=OPENAI_MODEL_NAME, temperature=0.1, response_format={"type": "json_object"}, messages=[{"role": "system", "content": "你是法律辅助分析模块。只基于 retrieved_evidence 归纳，不得编造法条、案号或事实。严格输出 JSON：{\"issues\":{\"favorable\":[],\"unfavorable\":[]},\"draft\":\"\"}"}, {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}])
            break
        except Exception as error:
            last_error = error
            if attempt == 0:
                time.sleep(0.2)
    else:
        raise last_error or RuntimeError("模型调用失败")
    result = json.loads(extract_json_text(completion.choices[0].message.content or ""))
    if not isinstance(result, dict) or not isinstance(result.get("issues"), dict) or not isinstance(result.get("draft"), str):
        raise ValueError("模型输出缺少 issues 或 draft 字段")
    if not all(isinstance(result["issues"].get(key, []), list) for key in ("favorable", "unfavorable")):
        raise ValueError("模型 issues 字段格式错误")
    return result


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", f"req_{uuid.uuid4().hex[:12]}")
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    REQUEST_COUNTER.labels(request.url.path, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(request.url.path).observe(time.perf_counter() - started)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = str(round((time.perf_counter() - started) * 1000, 2))
    return response


@app.get("/")
def health_check() -> Dict[str, str]:
    return {"message": "AI 法律案件辅助分析平台已启动", "version": app.version}


@app.get("/health/live")
def liveness() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
def readiness() -> Dict[str, str]:
    if not LAWS_DB_PATH.exists():
        raise HTTPException(status_code=503, detail="法律资料库不可用")
    try:
        corpus = load_laws_db()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=503, detail="法律资料库未通过版本和来源校验") from error
    return {"status": "ready", "corpus": corpus.get("metadata", {}).get("version", "unknown")}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/retrieval/search")
def retrieval_search(q: str = "押金退还", identity: dict[str, str] = Depends(current_identity)) -> Dict[str, Any]:
    db = load_laws_db()
    query_vector = embed_text(q)
    documents = db.get("laws", []) + db.get("cases", [])
    results = hybrid_search(q, documents, top_k=10, query_vector=query_vector)
    return {"query": q, "backend": "embedding+lexical" if query_vector else "lexical-fallback", "results": [{"id": item.document.get("id"), "title": item.document.get("title", item.document.get("case_title", "")), "bm25_score": round(item.bm25_score, 4), "vector_score": round(item.vector_score, 4), "rerank_score": round(item.rerank_score, 4)} for item in results]}


@app.post("/api/auth/token")
def create_token(payload: TokenRequest) -> Dict[str, str]:
    """演示用 token 签发端点；生产环境应接企业 SSO/OIDC。"""
    if os.getenv("REQUIRE_AUTH", "false").lower() == "true" and os.getenv("ALLOW_DEMO_TOKEN", "false").lower() != "true":
        raise HTTPException(status_code=404, detail="认证服务由企业身份提供商管理")
    return {"access_token": issue_token(payload.tenant_id, payload.user_id), "token_type": "bearer"}


def _save_analysis(response: AnalyzeResponse, identity: dict[str, str], case_id: str | None = None) -> str:
    with SessionLocal() as session:
        actual_case_id = case_id or f"case_{uuid.uuid4().hex[:16]}"
        if not case_id:
            session.add(CaseRecord(id=actual_case_id, tenant_id=identity["tenant_id"]))
        analysis_id = f"analysis_{uuid.uuid4().hex[:16]}"
        response.analysis_id = analysis_id
        session.add(AnalysisRecord(id=analysis_id, case_id=actual_case_id, tenant_id=identity["tenant_id"], request_id=response.request_id, status=response.status, output_json=response.model_dump(mode="json")))
        session.commit()
        return analysis_id


@app.get("/api/cases")
def list_cases(identity: dict[str, str] = Depends(current_identity)) -> List[Dict[str, Any]]:
    with SessionLocal() as session:
        records = session.scalars(select(CaseRecord).where(CaseRecord.tenant_id == identity["tenant_id"]).order_by(CaseRecord.created_at.desc()).limit(100)).all()
        return [{"case_id": item.id, "title": item.title, "created_at": item.created_at.isoformat()} for item in records]


@app.get("/api/cases/{case_id}")
def get_case(case_id: str, identity: dict[str, str] = Depends(current_identity)) -> Dict[str, Any]:
    with SessionLocal() as session:
        case = session.scalar(select(CaseRecord).where(CaseRecord.id == case_id, CaseRecord.tenant_id == identity["tenant_id"]))
        if not case:
            raise HTTPException(status_code=404, detail="案件不存在")
        analyses = session.scalars(select(AnalysisRecord).where(AnalysisRecord.case_id == case_id, AnalysisRecord.tenant_id == identity["tenant_id"]).order_by(AnalysisRecord.created_at.desc())).all()
        return {"case_id": case.id, "title": case.title, "analyses": [{"analysis_id": item.id, "status": item.status, "created_at": item.created_at.isoformat(), "request_id": item.request_id} for item in analyses]}


@app.post("/api/cases/{case_id}/reviews")
def review_analysis(case_id: str, payload: ReviewRequest, identity: dict[str, str] = Depends(current_identity)) -> Dict[str, str]:
    with SessionLocal() as session:
        record = session.scalar(select(AnalysisRecord).where(AnalysisRecord.case_id == case_id, AnalysisRecord.tenant_id == identity["tenant_id"]).order_by(AnalysisRecord.created_at.desc()))
        if not record:
            raise HTTPException(status_code=404, detail="案件分析记录不存在")
        review = ReviewRecord(id=f"review_{uuid.uuid4().hex[:16]}", analysis_id=record.id, tenant_id=identity["tenant_id"], reviewer=identity["user_id"], decision=payload.decision, comment=payload.comment)
        session.add(review)
        session.commit()
        return {"review_id": review.id, "status": payload.decision}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...), identity: dict[str, str] = Depends(current_identity)) -> Dict[str, Any]:
    if file.content_type not in {"text/plain", "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "image/png", "image/jpeg"}:
        raise HTTPException(status_code=415, detail="仅支持 TXT、PDF、DOCX、PNG、JPG")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="文件超过大小限制")
    extracted = ""
    if file.content_type == "text/plain":
        extracted = content.decode("utf-8", errors="replace")
    elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        import io
        document = Document(io.BytesIO(content))
        extracted = "\n".join(paragraph.text for paragraph in document.paragraphs)
    elif file.content_type == "application/pdf":
        try:
            from pypdf import PdfReader
            extracted = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages)
        except Exception:
            extracted = "PDF 文本层解析失败，需要 OCR 服务。"
    elif file.content_type.startswith("image/"):
        try:
            from PIL import Image
            import pytesseract
            extracted = pytesseract.image_to_string(Image.open(io.BytesIO(content)), lang=os.getenv("OCR_LANG", "eng+chi_sim"))
        except Exception:
            extracted = "图片 OCR 服务未配置或执行失败。"
    return {"document_id": f"doc_{uuid.uuid4().hex[:16]}", "tenant_id": identity["tenant_id"], "filename": file.filename, "content_type": file.content_type, "bytes": len(content), "extracted_text": redact_pii(extracted)[:20_000], "ocr_status": "COMPLETED" if extracted and "失败" not in extracted and "未配置" not in extracted else "PENDING"}


@app.get("/api/analyses/{analysis_id}/export", response_class=PlainTextResponse)
def export_analysis(analysis_id: str, identity: dict[str, str] = Depends(current_identity)) -> str:
    with SessionLocal() as session:
        record = session.scalar(select(AnalysisRecord).where(AnalysisRecord.id == analysis_id, AnalysisRecord.tenant_id == identity["tenant_id"]))
        if not record:
            raise HTTPException(status_code=404, detail="分析记录不存在")
        data = record.output_json
        return f"法律案件辅助分析报告\n请求 ID：{data.get('request_id')}\n状态：{data.get('status')}\n\n法律依据：\n- " + "\n- ".join(data.get("laws", [])) + f"\n\n诉讼方案初稿：\n{data.get('draft', '')}\n\n免责声明：{data.get('disclaimer', '')}"


@app.get("/api/analyses/{analysis_id}/export.docx")
def export_analysis_docx(analysis_id: str, identity: dict[str, str] = Depends(current_identity)) -> StreamingResponse:
    with SessionLocal() as session:
        record = session.scalar(select(AnalysisRecord).where(AnalysisRecord.id == analysis_id, AnalysisRecord.tenant_id == identity["tenant_id"]))
        if not record:
            raise HTTPException(status_code=404, detail="分析记录不存在")
        data = record.output_json
    document = Document()
    document.add_heading("法律案件辅助分析报告", level=1)
    document.add_paragraph(f"请求 ID：{data.get('request_id')}    状态：{data.get('status')}")
    document.add_heading("法律依据", level=2)
    for law in data.get("laws", []):
        document.add_paragraph(law, style="List Bullet")
    document.add_heading("诉讼方案初稿", level=2)
    document.add_paragraph(data.get("draft", ""))
    document.add_heading("待补充证据", level=2)
    for item in data.get("missing_evidence", []):
        document.add_paragraph(item, style="List Bullet")
    document.add_paragraph(data.get("disclaimer", ""))
    output = io.BytesIO()
    document.save(output)
    output.seek(0)
    return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers={"Content-Disposition": f'attachment; filename="legal-analysis-{analysis_id}.docx"'})


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_case(payload: AnalyzeRequest, request: Request, identity: dict[str, str] = Depends(current_identity), idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")) -> AnalyzeResponse:
    enforce_rate_limit(identity)
    if idempotency_key:
        cache_key = f"idem:{identity['tenant_id']}:{idempotency_key}"
        cached_value = CACHE.get(cache_key) if CACHE.url else None
        cached = (time.time(), cached_value) if cached_value else IDEMPOTENCY_CACHE.get(f"{identity['tenant_id']}:{idempotency_key}")
        if cached and (CACHE.url or cached[0] > time.time() - 3600):
            return AnalyzeResponse.model_validate(cached[1])
    request_id = getattr(request.state, "request_id", request.headers.get("X-Request-ID", f"req_{uuid.uuid4().hex[:12]}"))
    try:
        db = load_laws_db()
        context = retrieve_relevant_context(payload.case_description, payload.claims, db)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
        logger.exception("knowledge_base_unavailable request_id=%s", request_id)
        raise HTTPException(status_code=503, detail="法律资料库暂不可用") from error
    if len(payload.case_description) < 40:
        response = build_analysis(request_id, payload, context, status="NEEDS_MORE_INFO")
        _save_analysis(response, identity)
        if idempotency_key:
            value = response.model_dump(mode="json")
            if CACHE.url:
                CACHE.set(f"idem:{identity['tenant_id']}:{idempotency_key}", value)
            else:
                IDEMPOTENCY_CACHE[f"{identity['tenant_id']}:{idempotency_key}"] = (time.time(), value)
        return response
    try:
        response = build_analysis(request_id, payload, context, status="SUCCESS", model_result=call_llm_analysis(payload, context))
    except ValueError as error:
        logger.warning("invalid_model_output request_id=%s error=%s", request_id, error)
        response = build_analysis(request_id, payload, context, status="MODEL_OUTPUT_INVALID")
    except Exception as error:
        logger.warning("llm_degraded request_id=%s error_type=%s", request_id, type(error).__name__)
        response = build_analysis(request_id, payload, context, status="DEGRADED")
    _save_analysis(response, identity)
    if idempotency_key:
        value = response.model_dump(mode="json")
        if CACHE.url:
            CACHE.set(f"idem:{identity['tenant_id']}:{idempotency_key}", value)
        else:
            IDEMPOTENCY_CACHE[f"{identity['tenant_id']}:{idempotency_key}"] = (time.time(), value)
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
