"""Async client for the Drivvo private API."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

import aiohttp

from .const import (
    MODULE_CHECKLIST,
    MODULE_EXPENSE,
    MODULE_INCOME,
    MODULE_READING,
    MODULE_REFUELLING,
    MODULE_REMINDER,
    MODULE_ROUTE,
    MODULE_SERVICE,
)
from .models import (
    AccountConfig,
    Checklist,
    Expense,
    Income,
    Reading,
    Refuelling,
    Route,
    Service,
    VehicleData,
)

_LOGGER = logging.getLogger(__name__)

BASE_URL = "https://api.drivvo.com"

DISTANCE_UNITS = {1: "km", 2: "mi", 3: "h"}

# Every call is bounded: a poll makes eleven sequential requests, and a single
# stalled one would otherwise keep `async_setup_entry` pending forever.
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30)
LOGIN_URL = f"{BASE_URL}/autenticacao/login"


class DrivvoApiError(Exception):
    """Raised when the Drivvo API returns an unexpected response."""


class DrivvoAuthError(DrivvoApiError):
    """Raised when credentials are rejected (HTTP 401/403)."""


def _to_float(value: Any) -> float | None:
    """Coerce an API value to float, tolerating None and empty strings."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _default_headers() -> dict[str, str]:
    return {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "App-Platform": "Web",
        "App-Version": "19",
        "App-Version-Name": "14.0",
        "Content-Type": "application/json",
        "Origin": "https://web.drivvo.com",
        "Referer": "https://web.drivvo.com/",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
        ),
    }


class DrivvoApiClient:
    """Async wrapper over the Drivvo private API."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        email: str,
        password: str,
    ) -> None:
        self._session = session
        self._email = email
        # Drivvo's web API requires md5(password) in the "senha" field; not a security choice.
        self._password_md5 = hashlib.md5(
            password.encode("utf-8"), usedforsecurity=False
        ).hexdigest()
        self.token: str | None = None

    async def login(self) -> str:
        try:
            return await self._login_inner()
        except aiohttp.ClientError as err:
            raise DrivvoApiError("Connection to Drivvo failed") from err
        except TimeoutError as err:
            raise DrivvoApiError("Login timed out") from err

    async def _login_inner(self) -> str:
        async with self._session.post(
            LOGIN_URL,
            json={"email": self._email, "senha": self._password_md5},
            headers=_default_headers(),
            timeout=REQUEST_TIMEOUT,
        ) as resp:
            if resp.status in (401, 403):
                raise DrivvoAuthError(f"Login rejected: HTTP {resp.status}")
            if resp.status >= 400:
                raise DrivvoApiError(f"Login failed: HTTP {resp.status}")
            data = await resp.json()
            token = data.get("token")
            if not token:
                raise DrivvoApiError("Login response missing 'token'")
            self.token = token
            return token

    def _auth_headers(self) -> dict[str, str]:
        if not self.token:
            raise DrivvoApiError("Client is not authenticated; call login() first")
        h = _default_headers()
        h["x-token"] = self.token
        return h

    async def _get_json(self, path: str) -> Any:
        url = f"{BASE_URL}{path}"
        started = time.monotonic()
        try:
            return await self._get_json_inner(url, path)
        except aiohttp.ClientError as err:
            raise DrivvoApiError("Connection to Drivvo failed") from err
        except TimeoutError as err:
            raise DrivvoApiError(f"GET {path} timed out") from err
        finally:
            # Names the slow endpoint when a poll drags; a vehicle costs eleven
            # sequential requests.
            _LOGGER.debug("GET %s took %.2fs", path, time.monotonic() - started)

    async def _get_json_inner(self, url: str, path: str) -> Any:
        async with self._session.get(
            url, headers=self._auth_headers(), timeout=REQUEST_TIMEOUT
        ) as resp:
            if resp.status in (401, 403):
                raise DrivvoAuthError(f"Token rejected for {path}")
            if resp.status >= 400:
                raise DrivvoApiError(f"GET {path} failed: HTTP {resp.status}")
            return await resp.json()

    async def _get_json_optional(self, path: str, default: Any) -> Any:
        """GET a path the account may not have access to.

        Permission profiles can hide individual modules; a vehicle without
        reminders must still produce its refuelling sensors.
        """
        try:
            data = await self._get_json(path)
        except DrivvoAuthError:
            raise
        except DrivvoApiError as err:
            _LOGGER.debug("Optional endpoint %s unavailable: %s", path, err)
            return default
        return default if data is None else data

    async def _post_json(self, path: str, payload: dict[str, Any]) -> Any:
        try:
            return await self._post_json_inner(path, payload)
        except aiohttp.ClientError as err:
            raise DrivvoApiError("Connection to Drivvo failed") from err
        except TimeoutError as err:
            raise DrivvoApiError(f"POST {path} timed out") from err

    async def _post_json_inner(self, path: str, payload: dict[str, Any]) -> Any:
        url = f"{BASE_URL}{path}"
        async with self._session.post(
            url, json=payload, headers=self._auth_headers(), timeout=REQUEST_TIMEOUT
        ) as resp:
            if resp.status in (401, 403):
                raise DrivvoAuthError(f"Token rejected for {path}")
            if resp.status >= 400:
                body = (await resp.text())[:300]
                raise DrivvoApiError(f"POST {path} failed: HTTP {resp.status} {body}")
            if not resp.content_length:
                return None
            try:
                return await resp.json()
            except (aiohttp.ContentTypeError, ValueError):
                return None

    async def add_record(self, module: str, payload: dict[str, Any]) -> Any:
        """Create a record in one of the history modules."""
        return await self._post_json(f"/{module}", payload)

    async def get_lookup(self, path: str) -> list[dict[str, Any]]:
        """Fetch one of the account lookup tables (fuels, types, places...)."""
        return await self._get_json_optional(f"/{path}", [])

    async def _module(self, vehicle_id: int, module: str) -> list[dict[str, Any]]:
        """Fetch one history module, tolerating a module the account cannot see."""
        return await self._get_json_optional(f"/veiculo/{vehicle_id}/{module}/web", [])

    async def get_vehicles(self) -> list[dict[str, Any]]:
        return await self._get_json("/veiculo/web")

    async def get_vehicle(self, vehicle_id: int) -> VehicleData:
        # Sequential awaits (not asyncio.gather): once-per-day cadence makes parallelism
        # negligible, and short-circuiting on first failure is the right error semantic.
        detail = await self._get_json(f"/veiculo/{vehicle_id}")
        refuel_rows = await self._get_json(
            f"/veiculo/{vehicle_id}/{MODULE_REFUELLING}/web"
        )
        service_rows = await self._get_json(
            f"/veiculo/{vehicle_id}/{MODULE_SERVICE}/web"
        )
        expense_rows = await self._get_json(
            f"/veiculo/{vehicle_id}/{MODULE_EXPENSE}/web"
        )
        config = await self._get_json("/configuracao")

        # Modules a permission profile may hide, plus the account profile.
        income_rows = await self._module(vehicle_id, MODULE_INCOME)
        route_rows = await self._module(vehicle_id, MODULE_ROUTE)
        reading_rows = await self._module(vehicle_id, MODULE_READING)
        reminder_rows = await self._module(vehicle_id, MODULE_REMINDER)
        checklist_rows = await self._module(vehicle_id, MODULE_CHECKLIST)
        user = await self._get_json_optional("/usuario", {})

        config_row: dict[str, Any] | None = None
        if isinstance(config, list):
            config_row = config[0] if config and isinstance(config[0], dict) else None
        elif isinstance(config, dict):
            config_row = config
        account = AccountConfig.from_api(config_row)

        # The vehicle overrides the account-wide distance unit.
        unit_code = detail.get("unidade_distancia") or account.distance_unit_id

        if isinstance(user, list):
            user = user[0] if user else {}
        if not isinstance(user, dict):
            user = {}

        # Drivvo dates are YYYY-MM-DD HH:MM:SS — lex sort == chronological order.
        def newest(rows, model):
            return sorted(
                (model.from_api(r) for r in (rows or [])),
                key=lambda r: r.date or "",
                reverse=True,
            )

        return VehicleData(
            id=vehicle_id,
            name=detail.get("nome") or None,
            placa=detail.get("placa") or None,
            manufacturer=detail.get("marca", ""),
            model=detail.get("modelo", ""),
            distance_unit=DISTANCE_UNITS.get(unit_code, "km"),
            distance_unit_id=unit_code,
            currency=account.currency,
            config=account,
            year=detail.get("ano"),
            chassis=detail.get("chassi"),
            renavam=detail.get("renavam"),
            tank_volume=_to_float(detail.get("volume_tanque")),
            dual_fuel=detail.get("bicombustivel"),
            primary=detail.get("principal"),
            active=detail.get("ativo"),
            purchase_value=_to_float(detail.get("valor_compra")),
            sale_value=_to_float(detail.get("valor_venda")),
            note=detail.get("observacao"),
            fuel_type_id=detail.get("id_tipo_combustivel"),
            driver_license_expiry=user.get("cnh_validade"),
            refuellings=newest(refuel_rows, Refuelling),
            services=newest(service_rows, Service),
            expenses=newest(expense_rows, Expense),
            incomes=newest(income_rows, Income),
            routes=newest(route_rows, Route),
            readings=newest(reading_rows, Reading),
            checklists=newest(checklist_rows, Checklist),
            reminders_raw=reminder_rows or [],
        )
