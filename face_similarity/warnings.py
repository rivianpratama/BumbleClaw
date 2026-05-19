from __future__ import annotations

import warnings


def suppress_known_third_party_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message="`estimate` is deprecated.*",
        category=FutureWarning,
        module="insightface.utils.face_align",
    )
