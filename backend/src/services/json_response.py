from __future__ import annotations

import orjson
from fastapi.responses import ORJSONResponse as _BaseORJSONResponse


class ORJSONResponse(_BaseORJSONResponse):
    """orjson encoder with numpy passthrough. Lets routes return float32 ndarrays
    without `.tolist()` round-tripping — ~50x faster than the stdlib json path on
    large numeric payloads."""

    def render(self, content) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY)
