"""Tests for DrivvoApiClient."""

import pytest
from aiohttp import ClientSession

from custom_components.drivvo.api import (
    DrivvoApiClient,
    DrivvoApiError,
    DrivvoAuthError,
)
from tests.conftest import load


def _mock_empty_optional_modules(mock_aioresponse):
    """Describe the empty optional endpoints added in Drivvo 3.1.0."""
    for module in ("receita", "percurso", "leitura", "lembrete", "checklist"):
        mock_aioresponse.get(
            f"https://api.drivvo.com/veiculo/100/{module}/web", payload=[]
        )
    mock_aioresponse.get("https://api.drivvo.com/usuario", payload={})


@pytest.mark.asyncio
async def test_login_returns_token(mock_aioresponse):
    mock_aioresponse.post(
        "https://api.drivvo.com/autenticacao/login",
        payload=load("login_ok.json"),
        status=200,
    )

    async with ClientSession() as session:
        client = DrivvoApiClient(session, "user@example.com", "secret")
        token = await client.login()

    assert token == "fake-token-123"
    assert client.token == "fake-token-123"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_login_raises_on_auth_failure(mock_aioresponse, status):
    mock_aioresponse.post(
        "https://api.drivvo.com/autenticacao/login",
        status=status,
    )

    async with ClientSession() as session:
        client = DrivvoApiClient(session, "user@example.com", "wrong")
        with pytest.raises(DrivvoAuthError):
            await client.login()


@pytest.mark.asyncio
async def test_login_md5_hashes_password(mock_aioresponse):
    """Drivvo expects md5-hashed password in 'senha' field."""
    from yarl import URL

    mock_aioresponse.post(
        "https://api.drivvo.com/autenticacao/login",
        payload={"token": "t"},
        status=200,
    )

    async with ClientSession() as session:
        client = DrivvoApiClient(session, "user@example.com", "secret")
        await client.login()

    key = ("POST", URL("https://api.drivvo.com/autenticacao/login"))
    call = mock_aioresponse.requests[key][0]
    body = call.kwargs["json"]
    assert body["email"] == "user@example.com"
    # md5("secret") == "5ebe2294ecd0e0f08eab7690d2a6ee69"
    assert body["senha"] == "5ebe2294ecd0e0f08eab7690d2a6ee69"


@pytest.mark.asyncio
async def test_login_raises_when_response_missing_token(mock_aioresponse):
    mock_aioresponse.post(
        "https://api.drivvo.com/autenticacao/login",
        payload={},
        status=200,
    )

    async with ClientSession() as session:
        client = DrivvoApiClient(session, "user@example.com", "secret")
        with pytest.raises(DrivvoApiError, match="missing 'token'"):
            await client.login()


@pytest.mark.asyncio
async def test_get_vehicles(mock_aioresponse):
    mock_aioresponse.post(
        "https://api.drivvo.com/autenticacao/login",
        payload=load("login_ok.json"),
    )
    mock_aioresponse.get(
        "https://api.drivvo.com/veiculo/web",
        payload=load("vehicle_list.json"),
    )

    async with ClientSession() as session:
        client = DrivvoApiClient(session, "u", "p")
        await client.login()
        vehicles = await client.get_vehicles()

    assert len(vehicles) == 2
    assert vehicles[0]["id_veiculo"] == 100


@pytest.mark.asyncio
async def test_get_vehicle_aggregates_all_endpoints(mock_aioresponse):
    _mock_empty_optional_modules(mock_aioresponse)
    mock_aioresponse.post(
        "https://api.drivvo.com/autenticacao/login",
        payload=load("login_ok.json"),
    )
    mock_aioresponse.get(
        "https://api.drivvo.com/veiculo/100",
        payload=load("vehicle_detail.json"),
    )
    mock_aioresponse.get(
        "https://api.drivvo.com/veiculo/100/abastecimento/web",
        payload=load("refuellings.json"),
    )
    mock_aioresponse.get(
        "https://api.drivvo.com/veiculo/100/servico/web",
        payload=load("services.json"),
    )
    mock_aioresponse.get(
        "https://api.drivvo.com/veiculo/100/despesa/web",
        payload=load("expenses.json"),
    )
    mock_aioresponse.get(
        "https://api.drivvo.com/configuracao",
        payload=load("configuracao.json"),
    )

    async with ClientSession() as session:
        client = DrivvoApiClient(session, "u", "p")
        await client.login()
        vehicle = await client.get_vehicle(100)

    assert vehicle.id == 100
    assert vehicle.name == "My Car"
    assert vehicle.distance_unit == "km"
    assert vehicle.currency == "R$"
    assert len(vehicle.refuellings) == 2
    assert vehicle.refuellings[0].date == "2025-05-10 12:00:00"  # sorted desc
    assert len(vehicle.services) == 1
    assert len(vehicle.expenses) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.get_vehicles(),
        lambda c: c.get_vehicle(100),
    ],
)
async def test_get_requires_token(mock_aioresponse, call):
    async with ClientSession() as session:
        client = DrivvoApiClient(session, "u", "p")
        with pytest.raises(DrivvoApiError, match="not authenticated"):
            await call(client)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,expected",
    [
        (401, DrivvoAuthError),
        (403, DrivvoAuthError),
        (500, DrivvoApiError),
        (503, DrivvoApiError),
    ],
)
async def test_get_json_error_branches(mock_aioresponse, status, expected):
    # repeat: a 401/403 triggers one re-login before the error surfaces.
    mock_aioresponse.post(
        "https://api.drivvo.com/autenticacao/login",
        payload=load("login_ok.json"),
        repeat=True,
    )
    mock_aioresponse.get(
        "https://api.drivvo.com/veiculo/web",
        status=status,
        repeat=True,
    )

    async with ClientSession() as session:
        client = DrivvoApiClient(session, "u", "p")
        await client.login()
        with pytest.raises(expected):
            await client.get_vehicles()


@pytest.mark.asyncio
@pytest.mark.parametrize("config_payload", [[], {}, [{}]])
async def test_get_vehicle_handles_empty_or_malformed_currency_config(
    mock_aioresponse, config_payload
):
    _mock_empty_optional_modules(mock_aioresponse)
    mock_aioresponse.post(
        "https://api.drivvo.com/autenticacao/login",
        payload=load("login_ok.json"),
    )
    mock_aioresponse.get(
        "https://api.drivvo.com/veiculo/100",
        payload=load("vehicle_detail.json"),
    )
    mock_aioresponse.get(
        "https://api.drivvo.com/veiculo/100/abastecimento/web",
        payload=[],
    )
    mock_aioresponse.get(
        "https://api.drivvo.com/veiculo/100/servico/web",
        payload=[],
    )
    mock_aioresponse.get(
        "https://api.drivvo.com/veiculo/100/despesa/web",
        payload=[],
    )
    mock_aioresponse.get(
        "https://api.drivvo.com/configuracao",
        payload=config_payload,
    )

    async with ClientSession() as session:
        client = DrivvoApiClient(session, "u", "p")
        await client.login()
        vehicle = await client.get_vehicle(100)

    assert vehicle.currency is None


@pytest.mark.parametrize("operation", ["login", "get", "post"])
@pytest.mark.parametrize("failure", ["connection", "timeout", "server"])
async def test_transport_failures_are_api_errors(mock_aioresponse, operation, failure):
    from aiohttp import ClientConnectionError

    path = "/autenticacao/login" if operation == "login" else "/veiculo/web"
    mock = mock_aioresponse.get if operation == "get" else mock_aioresponse.post
    kwargs = (
        {"status": 503}
        if failure == "server"
        else {
            "exception": (
                ClientConnectionError("offline")
                if failure == "connection"
                else TimeoutError()
            )
        }
    )
    mock(f"https://api.drivvo.com{path}", **kwargs)
    async with ClientSession() as session:
        client = DrivvoApiClient(session, "u", "p")
        client.token = "token"
        with pytest.raises(DrivvoApiError) as exc:
            if operation == "login":
                await client.login()
            elif operation == "get":
                await client.get_vehicles()
            else:
                await client._post_json(path, {})
        assert not isinstance(exc.value, DrivvoAuthError)
