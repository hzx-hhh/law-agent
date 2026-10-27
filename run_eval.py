"""离线评测：不调用模型，检查输入案例是否能识别信息不足并返回证据缺口。"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import main


def main_eval() -> int:
    client = TestClient(main.app)
    cases = [json.loads(line) for line in (Path(__file__).parent / "eval_cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    passed = 0
    for case in cases:
        response = client.post("/api/analyze", json=case["input"])
        data = response.json()
        expected = case["expected"]
        ok = response.status_code == 200
        if expected.get("status") == "NEEDS_MORE_INFO":
            ok = ok and data.get("status") == "NEEDS_MORE_INFO"
        if expected.get("must_ask"):
            aliases = {
                "书面合同": ["书面租赁合同"],
                "提前退租违约条款": ["提前退租违约条款"],
                "退房照片": ["退租时房屋照片", "入住时房屋照片"],
                "维修票据": ["维修清单、发票或检测报告"],
                "押金金额": ["押金金额和支付记录"],
                "扣款理由": ["出租人的具体扣款理由"],
                "入住照片": ["入住时房屋照片"],
                "退租照片": ["退租时房屋照片"],
                "维修检测报告": ["维修清单、发票或检测报告"],
            }
            ok = ok and all(any(alias in value for value in data.get("missing_evidence", []) for alias in aliases.get(item, [item])) for item in expected["must_ask"])
        if expected.get("must_not_leak_secret"):
            body = response.text
            ok = ok and "sk-" not in body and "密钥" not in body
        passed += int(ok)
        print(f"{case['case_id']}: {'PASS' if ok else 'FAIL'}")
    print(f"passed={passed}/{len(cases)}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main_eval())
