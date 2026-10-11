"""F05: a stuck engine stage becomes a recoverable timeout, and its worker is released."""
from app import main


def test_timeout_is_classified_as_retryable_timeout():
    status, code, msg = main._classify_error(main.EngineTimeout(main.MSG_TIMEOUT))
    assert (status, code) == (504, "timeout")
    assert "smaller or clearer photo" in msg


def test_kill_pool_releases_processes_and_retires_pool():
    calls = []

    class Proc:
        def kill(self):
            calls.append("kill")

    class Pool:
        _processes = {1: Proc()}

        def shutdown(self, wait=True, cancel_futures=False):
            calls.append(("shutdown", wait, cancel_futures))

    pool = Pool()
    main._kill_pool_for(pool)
    assert "kill" in calls and ("shutdown", False, True) in calls
    assert pool in main._KILLED_POOLS
