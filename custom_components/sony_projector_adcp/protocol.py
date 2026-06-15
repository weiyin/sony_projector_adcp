"""Sony ADCP Protocol Handler."""
import asyncio
import hashlib
import logging
from typing import Optional, Tuple

_LOGGER = logging.getLogger(__name__)

NEWLINE = "\r\n"
ENCODING = "ascii"
TIMEOUT = 10
CLOSE_TIMEOUT = 2


class SonyProjectorADCP:
    """Handle ADCP protocol communication with Sony projector.

    Sony's ADCP server closes idle TCP sockets after roughly 30 seconds,
    so we open a fresh connection for each command rather than holding
    one open. The lock serializes concurrent callers.
    """

    def __init__(self, host: str, port: int, password: str = "", use_auth: bool = True):
        self.host = host
        self.port = port
        self.password = password
        self.use_auth = use_auth
        self._lock = asyncio.Lock()

    async def _open(self) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Open a TCP connection and complete the ADCP auth handshake."""
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port),
            timeout=TIMEOUT,
        )
        try:
            data = await asyncio.wait_for(
                reader.readuntil(NEWLINE.encode(ENCODING)),
                timeout=TIMEOUT,
            )
            auth_response = data.decode(ENCODING).strip()

            if auth_response.startswith("PJLINK") or not self.use_auth:
                if auth_response == "NOKEY":
                    _LOGGER.debug("Authentication disabled on projector")
                    return reader, writer

            if self.use_auth and auth_response and auth_response != "NOKEY":
                hash_input = f"{auth_response}{self.password}"
                hash_result = hashlib.sha256(hash_input.encode()).hexdigest()

                writer.write(f"{hash_result}{NEWLINE}".encode(ENCODING))
                await writer.drain()

                ack = await asyncio.wait_for(
                    reader.readuntil(NEWLINE.encode(ENCODING)),
                    timeout=TIMEOUT,
                )
                auth_result = ack.decode(ENCODING).strip()
                if auth_result != "OK":
                    raise ConnectionError(f"ADCP authentication failed: {auth_result}")

            return reader, writer
        except Exception:
            await self._close(writer)
            raise

    async def _close(self, writer: Optional[asyncio.StreamWriter]) -> None:
        """Close a writer with a bounded wait — a half-open socket must not wedge us."""
        if writer is None:
            return
        try:
            writer.close()
            await asyncio.wait_for(writer.wait_closed(), timeout=CLOSE_TIMEOUT)
        except (asyncio.TimeoutError, Exception) as e:
            _LOGGER.debug("Error/timeout closing connection: %s", e)

    async def connect(self) -> bool:
        """Verify we can reach the projector. Used at config-entry setup."""
        writer = None
        try:
            _, writer = await self._open()
        except asyncio.TimeoutError:
            _LOGGER.error("Timeout connecting to projector")
            return False
        except Exception as e:
            _LOGGER.error("Error connecting to projector: %s", e)
            return False
        finally:
            await self._close(writer)
        _LOGGER.info("Connected to Sony projector at %s:%s", self.host, self.port)
        return True

    async def disconnect(self):
        """No-op: connections are per-command and closed immediately."""
        return

    async def send_command(self, command: str) -> Optional[str]:
        """Open a connection, send a command, read the response, then close."""
        async with self._lock:
            writer = None
            try:
                reader, writer = await self._open()

                writer.write(f"{command}{NEWLINE}".encode(ENCODING))
                await writer.drain()
                _LOGGER.debug("Sent command: %s", command)

                data = await asyncio.wait_for(
                    reader.readuntil(NEWLINE.encode(ENCODING)),
                    timeout=TIMEOUT,
                )
                response = data.decode(ENCODING).strip()
                _LOGGER.debug("Received response: %s", response)

                if response.startswith("err_"):
                    _LOGGER.error("Command error: %s for command: %s", response, command)
                    return None

                return response
            except asyncio.TimeoutError:
                _LOGGER.error("Timeout sending command: %s", command)
                return None
            except Exception as e:
                _LOGGER.error("Error sending command %s: %s", command, e)
                return None
            finally:
                await self._close(writer)

    async def get_power_status(self) -> Optional[str]:
        """Get the current power status."""
        response = await self.send_command("power_status ?")
        if response and response.startswith('"') and response.endswith('"'):
            return response.strip('"')
        return None

    async def set_power(self, state: bool) -> bool:
        """Set power on or off."""
        command = 'power "on"' if state else 'power "off"'
        response = await self.send_command(command)
        return response == "ok"

    async def get_input(self) -> Optional[str]:
        """Get current input source."""
        response = await self.send_command("input ?")
        if response and response.startswith('"') and response.endswith('"'):
            return response.strip('"')
        return None

    async def set_input(self, source: str) -> bool:
        """Set input source."""
        command = f'input "{source}"'
        response = await self.send_command(command)
        return response == "ok"

    async def get_blank_status(self) -> Optional[bool]:
        """Get video muting status."""
        response = await self.send_command("blank ?")
        if response and response.startswith('"') and response.endswith('"'):
            return response.strip('"') == "on"
        return None

    async def set_blank(self, state: bool) -> bool:
        """Set video muting."""
        command = 'blank "on"' if state else 'blank "off"'
        response = await self.send_command(command)
        return response == "ok"

    async def get_picture_mode(self) -> Optional[str]:
        """Get current picture mode."""
        response = await self.send_command("picture_mode ?")
        if response and response.startswith('"') and response.endswith('"'):
            return response.strip('"')
        return None

    async def set_picture_mode(self, mode: str) -> bool:
        """Set picture mode."""
        command = f'picture_mode "{mode}"'
        response = await self.send_command(command)
        return response == "ok"

    async def get_numeric_value(self, parameter: str) -> Optional[int]:
        """Get a numeric parameter value."""
        response = await self.send_command(f"{parameter} ?")
        if response and response.isdigit():
            return int(response)
        # Handle negative numbers
        if response and response.lstrip('-').isdigit():
            return int(response)
        return None

    async def set_numeric_value(self, parameter: str, value: int) -> bool:
        """Set a numeric parameter value."""
        command = f"{parameter} {value}"
        response = await self.send_command(command)
        return response == "ok"

    async def send_key(self, key: str) -> bool:
        """Send a remote control key command."""
        command = f'key "{key}"'
        response = await self.send_command(command)
        return response == "ok"

    async def get_reality_creation(self) -> Optional[str]:
        """Get Reality Creation status."""
        response = await self.send_command("real_cre ?")
        if response and response.startswith('"') and response.endswith('"'):
            return response.strip('"')
        return None

    async def set_reality_creation(self, state: str) -> bool:
        """Set Reality Creation on/off."""
        command = f'real_cre "{state}"'
        response = await self.send_command(command)
        return response == "ok"
