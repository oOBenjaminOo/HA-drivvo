"""API client coverage for the endpoints added in this release.

Uses a hand-rolled session stub rather than aioresponses, which does not yet
support aiohttp 3.14 (the version Home Assistant pins).
"""

import asyncio

import pytest

from custom_components.drivvo.api import (
    DrivvoApiClient,
    DrivvoApiError,
    DrivvoAuthError,
)


class _FakeResponse:
    def __init__(self, status=200, payload=None, body=""):
        self.status = status
        self._payload = payload
        self._body = body
        self.content_length = len(body) if body else (1 if payload is not None else 0)

    async def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    async def text(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Routes requests through a caller supplied table of responses."""

    def __init__(self, routes: dict, default=None):
        self.routes = routes
        self.default = default
        self.calls: list[tuple[str, str, dict | None, dict]] = []

    def _handle(self, method, url, json=None, headers=None):
        self.calls.append((method, url, json, headers or {}))
        path = url.replace("https://api.drivvo.com", "")
        result = self.routes.get((method, path), self.default)
        if result is None:
            return _FakeResponse(404, body="not found")
        if callable(result):
            return result()
        return result

    def get(self, url, headers=None, timeout=None):
        return self._handle("GET", url, None, headers)

    def post(self, url, json=None, headers=None, timeout=None):
        return self._handle("POST", url, json, headers)


def _client(session):
    client = DrivvoApiClient(session, "user@example.com", "secret")
    client.token = "tok"
    return client


@pytest.mark.asyncio
async def test_add_record_posts_to_the_module_root():
    session = _FakeSession(
        {("POST", "/abastecimento"): _FakeResponse(200, {"id_abastecimento": 9})}
    )
    client = _client(session)

    result = await client.add_record("abastecimento", {"odometro": 1000})

    assert result == {"id_abastecimento": 9}
    method, url, payload, headers = session.calls[0]
    assert (method, url) == ("POST", "https://api.drivvo.com/abastecimento")
    assert payload == {"odometro": 1000}
    assert headers["x-token"] == "tok"


@pytest.mark.asyncio
async def test_add_record_surfaces_the_server_message():
    session = _FakeSession(
        {("POST", "/servico"): _FakeResponse(422, body="invalid odometro")}
    )
    client = _client(session)

    with pytest.raises(DrivvoApiError, match="422"):
        await client.add_record("servico", {})


_LOGIN = ("POST", "/autenticacao/login")


def _calls_to(session, path):
    return [c for c in session.calls if c[1] == f"https://api.drivvo.com{path}"]


def _rejects_token(session, rejected, ok):
    """Answer 401 while the request carries the rejected token, else `ok`."""

    def respond():
        if session.calls[-1][3].get("x-token") == rejected:
            return _FakeResponse(401)
        return ok

    return respond


@pytest.mark.asyncio
async def test_add_record_raises_auth_error_on_rejected_token():
    session = _FakeSession(
        {
            ("POST", "/despesa"): _FakeResponse(401),
            _LOGIN: _FakeResponse(200, {"token": "fresh"}),
        }
    )
    client = _client(session)

    with pytest.raises(DrivvoAuthError):
        await client.add_record("despesa", {})


@pytest.mark.asyncio
async def test_expired_token_is_renewed_and_the_read_retried():
    session = _FakeSession({_LOGIN: _FakeResponse(200, {"token": "fresh"})})
    session.routes[("GET", "/veiculo/web")] = _rejects_token(
        session, "tok", _FakeResponse(200, [{"id_veiculo": 1}])
    )
    client = _client(session)

    assert await client.get_vehicles() == [{"id_veiculo": 1}]
    assert [(c[0], c[1].removeprefix("https://api.drivvo.com")) for c in session.calls] == [
        ("GET", "/veiculo/web"),
        _LOGIN,
        ("GET", "/veiculo/web"),
    ]
    assert session.calls[-1][3]["x-token"] == "fresh"
    assert client.token == "fresh"


@pytest.mark.asyncio
async def test_expired_token_is_renewed_and_the_write_retried():
    session = _FakeSession({_LOGIN: _FakeResponse(200, {"token": "fresh"})})
    session.routes[("POST", "/combustivel")] = _rejects_token(
        session, "tok", _FakeResponse(200, {"id_abastecimento": 9})
    )
    client = _client(session)

    assert await client.add_record("combustivel", {"odometro": 1}) == {
        "id_abastecimento": 9
    }
    retried = _calls_to(session, "/combustivel")
    assert len(retried) == 2
    assert retried[1][2] == {"odometro": 1}
    assert retried[1][3]["x-token"] == "fresh"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_second_rejection_raises_without_further_retry(status):
    session = _FakeSession(
        {
            ("GET", "/veiculo/web"): _FakeResponse(status),
            _LOGIN: _FakeResponse(200, {"token": "fresh"}),
        }
    )
    client = _client(session)

    with pytest.raises(DrivvoAuthError):
        await client.get_vehicles()
    assert len(_calls_to(session, "/veiculo/web")) == 2
    assert len(_calls_to(session, "/autenticacao/login")) == 1


@pytest.mark.asyncio
async def test_rejected_relogin_raises_auth_error():
    session = _FakeSession(
        {("GET", "/veiculo/web"): _FakeResponse(401), _LOGIN: _FakeResponse(401)}
    )
    client = _client(session)

    with pytest.raises(DrivvoAuthError):
        await client.get_vehicles()
    assert len(_calls_to(session, "/veiculo/web")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 500])
async def test_other_errors_are_not_retried(status):
    session = _FakeSession({("GET", "/veiculo/web"): _FakeResponse(status)})
    client = _client(session)

    with pytest.raises(DrivvoApiError) as exc:
        await client.get_vehicles()
    assert not isinstance(exc.value, DrivvoAuthError)
    assert len(session.calls) == 1


class _SlowResponse(_FakeResponse):
    async def __aenter__(self):
        await asyncio.sleep(0)
        return self


@pytest.mark.asyncio
async def test_concurrent_rejections_share_a_single_login():
    session = _FakeSession({_LOGIN: _SlowResponse(200, {"token": "fresh"})})
    for path in ("/veiculo/1", "/veiculo/2"):
        session.routes[("GET", path)] = _rejects_token(
            session, "tok", _FakeResponse(200, {"path": path})
        )
    client = _client(session)

    results = await asyncio.gather(
        client._get_json("/veiculo/1"), client._get_json("/veiculo/2")
    )

    assert results == [{"path": "/veiculo/1"}, {"path": "/veiculo/2"}]
    assert len(_calls_to(session, "/autenticacao/login")) == 1


@pytest.mark.asyncio
async def test_optional_module_missing_does_not_break_the_poll():
    """A permission profile can hide a module; the rest must still load."""
    session = _FakeSession({("GET", "/veiculo/1/lembrete/web"): _FakeResponse(500)})
    client = _client(session)

    assert await client._module(1, "lembrete") == []


@pytest.mark.asyncio
async def test_optional_module_still_propagates_auth_failure():
    session = _FakeSession(
        {
            ("GET", "/veiculo/1/lembrete/web"): _FakeResponse(403),
            _LOGIN: _FakeResponse(200, {"token": "fresh"}),
        }
    )
    client = _client(session)

    with pytest.raises(DrivvoAuthError):
        await client._module(1, "lembrete")


@pytest.mark.asyncio
async def test_get_vehicle_reads_units_and_every_module():
    routes = {
        ("GET", "/veiculo/1"): _FakeResponse(
            200,
            {
                "id_veiculo": 1,
                "nome": "Car",
                "marca": "VW",
                "modelo": "Gol",
                "unidade_distancia": 2,
                "volume_tanque": 55,
                "ano": 2019,
                "id_tipo_combustivel": 1,
            },
        ),
        ("GET", "/veiculo/1/abastecimento/web"): _FakeResponse(200, []),
        ("GET", "/veiculo/1/servico/web"): _FakeResponse(200, []),
        ("GET", "/veiculo/1/despesa/web"): _FakeResponse(200, []),
        ("GET", "/configuracao"): _FakeResponse(
            200,
            [
                {
                    "formato_valor": "USD",
                    "unidade_distancia": 1,
                    "unidade_medida": 2,
                    "eficiencia_combustivel": 8,
                }
            ],
        ),
        ("GET", "/veiculo/1/receita/web"): _FakeResponse(200, []),
        ("GET", "/veiculo/1/percurso/web"): _FakeResponse(200, []),
        ("GET", "/veiculo/1/leitura/web"): _FakeResponse(200, []),
        ("GET", "/veiculo/1/lembrete/web"): _FakeResponse(200, [{"id_lembrete": 3}]),
        ("GET", "/veiculo/1/checklist/web"): _FakeResponse(200, []),
        ("GET", "/usuario"): _FakeResponse(200, {"cnh_validade": "2030-01-01"}),
    }
    client = _client(_FakeSession(routes))

    data = await client.get_vehicle(1)

    # The vehicle's own unit wins over the account-wide one.
    assert data.distance_unit == "mi"
    assert data.currency == "USD"
    assert data.volume_unit == "gal"
    assert data.efficiency_unit == "mpg"
    assert data.tank_volume == 55
    assert data.year == 2019
    assert data.driver_license_expiry == "2030-01-01"
    assert data.reminders_raw == [{"id_lembrete": 3}]


@pytest.mark.asyncio
async def test_get_lookup_returns_empty_when_unavailable():
    client = _client(_FakeSession({("GET", "/tipoServico"): _FakeResponse(500)}))
    assert await client.get_lookup("tipoServico") == []


class _HangingResponse(_FakeResponse):
    async def __aenter__(self):
        raise TimeoutError


@pytest.mark.asyncio
async def test_a_stalled_request_becomes_an_api_error():
    """A stalled endpoint must not keep `async_setup_entry` pending forever."""
    session = _FakeSession({("GET", "/veiculo/1"): _HangingResponse()})
    client = _client(session)

    with pytest.raises(DrivvoApiError, match="timed out"):
        await client._get_json("/veiculo/1")


@pytest.mark.asyncio
async def test_a_stalled_optional_module_degrades_to_empty():
    session = _FakeSession({("GET", "/veiculo/1/lembrete/web"): _HangingResponse()})
    client = _client(session)

    assert await client._module(1, "lembrete") == []


@pytest.mark.asyncio
async def test_a_stalled_write_becomes_an_api_error():
    session = _FakeSession({("POST", "/leitura"): _HangingResponse()})
    client = _client(session)

    with pytest.raises(DrivvoApiError, match="timed out"):
        await client.add_record("leitura", {})


@pytest.mark.asyncio
async def test_a_stalled_login_becomes_an_api_error():
    session = _FakeSession({("POST", "/autenticacao/login"): _HangingResponse()})
    client = DrivvoApiClient(session, "user@example.com", "secret")

    with pytest.raises(DrivvoApiError, match="timed out"):
        await client.login()
