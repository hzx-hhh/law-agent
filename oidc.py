"""OIDC/JWKS verification helper. Local HS256 demo tokens remain supported by main.py."""
from __future__ import annotations

import os
from typing import Any


class OIDCVerifier:
    def __init__(self) -> None:
        self.issuer = os.getenv("OIDC_ISSUER")
        self.audience = os.getenv("OIDC_AUDIENCE")

    def enabled(self) -> bool:
        return bool(self.issuer and self.audience)

    def verify(self, token: str) -> dict[str, Any]:
        if not self.enabled():
            raise RuntimeError("OIDC_ISSUER/OIDC_AUDIENCE 未配置")
        try:
            from jwt import PyJWKClient, decode
            jwks = PyJWKClient(f"{self.issuer.rstrip('/')}/.well-known/jwks.json")
            signing_key = jwks.get_signing_key_from_jwt(token).key
            return decode(token, signing_key, algorithms=["RS256"], audience=self.audience, issuer=self.issuer)
        except Exception as error:
            raise ValueError("OIDC token 校验失败") from error
