from __future__ import annotations

import pytest

from scripts.check_registry_status import (
    ACTIVE,
    RegistryStatusError,
    wait_for_active,
)


def test_pending_then_active_is_a_success() -> None:
    states = iter(
        [
            (False, "missing"),
            (True, "NodeVersionStatusPending"),
            (True, ACTIVE),
        ]
    )
    sleeps: list[float] = []

    def lookup(*_args, **_kwargs):
        return next(states)

    assert wait_for_active(
        "indextts25-t8",
        "0.12.0",
        attempts=3,
        interval=0.5,
        lookup=lookup,
        sleep=sleeps.append,
    ) == ACTIVE
    assert sleeps == [0.5, 0.5]


@pytest.mark.parametrize(
    "status", ["NodeVersionStatusFlagged", "NodeVersionStatusBanned"]
)
def test_security_rejection_fails_immediately(status: str) -> None:
    with pytest.raises(RegistryStatusError, match=status):
        wait_for_active(
            "indextts25-t8",
            "0.12.0",
            attempts=10,
            lookup=lambda *_args, **_kwargs: (True, status),
            sleep=lambda _seconds: (_ for _ in ()).throw(AssertionError("must not sleep")),
        )


def test_pending_is_never_reported_as_release_success() -> None:
    with pytest.raises(RegistryStatusError, match="last status=NodeVersionStatusPending"):
        wait_for_active(
            "indextts25-t8",
            "0.12.0",
            attempts=2,
            interval=0,
            lookup=lambda *_args, **_kwargs: (
                True,
                "NodeVersionStatusPending",
            ),
            sleep=lambda _seconds: None,
        )
