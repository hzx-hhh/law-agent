# 企业化升级设计

## 请求链路

```text
API Gateway/鉴权
  -> 输入校验与脱敏
  -> 案件事实抽取
  -> 混合检索（BM25 + 向量 + 重排）
  -> 证据包构建
  -> LLM Gateway（超时/重试/预算）
  -> JSON Schema 与引用校验
  -> 风险检测
  -> 人工审核
  -> 文书导出
```

本次实现完成了输入校验、可解释检索、证据包、结构化模型输出校验、降级状态、请求追踪和健康检查。数据库、向量库、鉴权、人工审核工作台和正式文书导出仍属于下一阶段。

## 第三阶段实现

- `retrieval.py` 提供 BM25/词法召回、embedding 余弦相似度和 rerank 分数；配置 `EMBEDDING_API_KEY` 后启用 embedding，否则自动回退词法检索。
- `pgvector_schema_sql()` 和 `pgvector_query_sql()` 提供 PostgreSQL 向量列、IVFFlat 索引和租户过滤查询模板。
- `redis_backend.py` 在配置 `REDIS_URL` 后使用 Redis 做限流和幂等缓存，未配置时使用本地内存回退。
- `migrations/` 和 `alembic.ini` 提供初始数据库迁移。
- `oidc.py` 支持通过 `OIDC_ISSUER`、`OIDC_AUDIENCE` 验证 JWKS；未配置时保留本地 HS256 演示令牌。
- 上传接口已支持 TXT、DOCX、PDF 文本层解析，以及通过 Tesseract 的图片 OCR。
- `CASE_WORKBENCH.html` 提供案件列表、分析版本、审核按钮和 DOCX 导出入口。
- `grafana-dashboard.json` 提供请求速率、P95 延迟、5xx 和分析状态面板。
- `eval_cases_generated.jsonl` 提供 50 条合成标注种子数据；投入生产前应由法律专业人员替换为脱敏真实样本并完成双人复核。

## 故障设计

系统把失败当作一等场景：模型异常返回 `DEGRADED` 或 `MODEL_OUTPUT_INVALID`，资料不足返回 `NEEDS_MORE_INFO`，跨租户资源返回 404，文件类型/大小错误返回 415/413，资料库版本缺失会阻断加载。完整复盘见 `BAD_CASES.md`，而 `failure_cases.jsonl` 可作为后续故障注入和 Chaos Test 的输入目录。

## 关键设计决策

1. 模型只能使用检索上下文做归纳，法条和案例名称由系统证据包提供。
2. 本地降级结果明确标记为 `DEGRADED`，不会伪装成模型成功结果。
3. 资料库版本和案例是否为合成数据会随响应返回，便于审计。
4. 用户输入被视为不可信数据，不能覆盖系统规则或读取环境变量。
5. 任何正式法律结论都需要人工审核。
