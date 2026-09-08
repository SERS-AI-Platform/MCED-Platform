import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from psycopg2 import sql

REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeConnection:
    def set_client_encoding(self, encoding: str) -> None:
        del encoding


class _FakeDatabaseCursor:
    def __init__(self) -> None:
        self.executions: list[tuple[sql.Composable | str, tuple[str, ...] | None]] = []

    def execute(
        self,
        query: sql.Composable | str,
        parameters: tuple[str, ...] | None = None,
    ) -> None:
        self.executions.append((query, parameters))

    def fetchone(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeDatabaseConnection:
    def __init__(self) -> None:
        self.cursor_instance = _FakeDatabaseCursor()

    def set_isolation_level(self, level: int) -> None:
        del level

    def cursor(self) -> _FakeDatabaseCursor:
        return self.cursor_instance

    def close(self) -> None:
        return None


def _load_script(name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def upload_script() -> ModuleType:
    return _load_script("upload_to_postgres_credentials", "scripts/db/upload_to_postgres.py")


@pytest.fixture
def create_script() -> ModuleType:
    return _load_script("db_create_credentials", "scripts/db/db_create.py")


@pytest.fixture
def export_script() -> ModuleType:
    return _load_script(
        "export_clinical_unified_credentials",
        "scripts/db/clinical_unified/export_clinical_unified.py",
    )


def test_database_scripts_have_no_password_fallback() -> None:
    paths = [
        REPO_ROOT / "scripts/db/upload_to_postgres.py",
        REPO_ROOT / "scripts/db/db_create.py",
        REPO_ROOT / "scripts/db/clinical_unified/export_clinical_unified.py",
    ]

    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert 'os.environ.get("PGPASSWORD",' not in source
        assert "os.environ.get('PGPASSWORD'," not in source


def test_upload_fails_before_network_when_password_missing(
    upload_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PGPASSWORD", raising=False)
    monkeypatch.setattr(upload_script.sys, "argv", ["upload_to_postgres.py"])

    network_calls: list[bool] = []
    monkeypatch.setattr(
        upload_script.psycopg2,
        "connect",
        lambda **_: network_calls.append(True),
    )

    with pytest.raises(SystemExit, match="password"):
        upload_script.main()

    assert network_calls == []


def test_upload_accepts_cli_and_environment_password(
    upload_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PGPASSWORD", raising=False)
    assert upload_script.resolve_password("cli-secret") == "cli-secret"

    monkeypatch.setenv("PGPASSWORD", "environment-secret")
    assert upload_script.resolve_password(None) == "environment-secret"


def test_database_name_is_one_composed_identifier(
    upload_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malicious_name = 'safe"; DROP DATABASE prod; --'
    connection = _FakeDatabaseConnection()
    monkeypatch.setattr(upload_script, "connect_pg", lambda **_: connection)

    upload_script.ensure_database(
        "localhost", 5432, "postgres", "environment-secret", malicious_name
    )

    create_query, create_parameters = connection.cursor_instance.executions[1]
    assert isinstance(create_query, sql.Composed)
    assert create_parameters is None
    parts = create_query._wrapped
    assert len(parts) == 2
    assert isinstance(parts[1], sql.Identifier)
    assert parts[1]._wrapped == (malicious_name,)


def test_db_create_fails_before_network_when_password_missing(
    create_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PGPASSWORD", raising=False)
    network_calls: list[bool] = []
    monkeypatch.setattr(
        create_script.psycopg2,
        "connect",
        lambda **_: network_calls.append(True),
    )

    with pytest.raises(SystemExit, match="password"):
        create_script.get_connection()

    assert network_calls == []


def test_db_create_accepts_cli_and_environment_password(
    create_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def fake_connect(**kwargs: str) -> _FakeConnection:
        captured["password"] = kwargs["password"]
        return _FakeConnection()

    monkeypatch.setattr(create_script.psycopg2, "connect", fake_connect)
    monkeypatch.delenv("PGPASSWORD", raising=False)
    create_script.get_connection("cli-secret")
    assert captured["password"] == "cli-secret"

    monkeypatch.setenv("PGPASSWORD", "environment-secret")
    create_script.get_connection()
    assert captured["password"] == "environment-secret"


def test_clinical_export_fails_before_network_when_password_missing(
    export_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PGPASSWORD", raising=False)
    monkeypatch.delenv("CLINICAL_DB_URL", raising=False)
    network_calls: list[bool] = []
    monkeypatch.setattr(
        export_script,
        "create_engine",
        lambda *_: network_calls.append(True),
    )

    with pytest.raises(SystemExit, match="password"):
        export_script.main()

    assert network_calls == []


def test_clinical_export_accepts_database_url_or_environment_password(
    export_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PGPASSWORD", raising=False)
    monkeypatch.setenv("CLINICAL_DB_URL", "postgresql://configured")
    assert str(export_script.resolve_db_url()) == "postgresql://configured"

    monkeypatch.delenv("CLINICAL_DB_URL", raising=False)
    monkeypatch.setenv("PGPASSWORD", "environment-secret")
    url = export_script.resolve_db_url()
    assert url.password == "environment-secret"
