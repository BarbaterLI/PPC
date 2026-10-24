"""DRM module is used to handle DRM operations with clock skew correction.
Currently the only DRM operation is generating the Sec-MS-GEC token value
used in all API requests to Microsoft Edge's online text-to-speech service.

Clock skew state lives in an injectable :class:`ClockSkew` instance instead
of a class-level mutable attribute.
"""

import hashlib
import secrets
from datetime import datetime as dt
from datetime import timezone as tz
from typing import Dict, Optional

import aiohttp

from .constants import TRUSTED_CLIENT_TOKEN
from .exceptions import SkewAdjustmentError

WIN_EPOCH = 11644473600
S_TO_NS = 1e9


class ClockSkew:
    """Clock skew state, injectable into :class:`DRM`.

    Holds the accumulated skew (in seconds) between the local system
    clock and the server clock, as learned from 403 responses.
    """

    def __init__(self) -> None:
        self._seconds: float = 0.0

    @property
    def seconds(self) -> float:
        return self._seconds

    def adjust(self, skew_seconds: float) -> None:
        """Adjust the clock skew in seconds in case the system clock is off."""
        self._seconds += skew_seconds

    def get_unix_timestamp(self) -> float:
        """Gets the current timestamp in Unix format with clock skew correction."""
        return dt.now(tz.utc).timestamp() + self._seconds


# Shared default skew state: 403-driven skew corrections learned by one
# request keep applying to all subsequent requests in this process
# (same behavior as the former class-level attribute).
DEFAULT_CLOCK_SKEW = ClockSkew()


class DRM:
    """
    Class to handle DRM operations with clock skew correction.
    """

    def __init__(self, clock_skew: Optional[ClockSkew] = None) -> None:
        self._clock_skew = clock_skew if clock_skew is not None else DEFAULT_CLOCK_SKEW

    def adj_clock_skew_seconds(self, skew_seconds: float) -> None:
        """
        Adjust the clock skew in seconds in case the system clock is off.

        This method updates the injected :class:`ClockSkew` instance to
        the specified number of seconds.

        Args:
            skew_seconds (float): The number of seconds to adjust the clock skew to.

        Returns:
            None
        """
        self._clock_skew.adjust(skew_seconds)

    def get_unix_timestamp(self) -> float:
        """
        Gets the current timestamp in Unix format with clock skew correction.

        Returns:
            float: The current timestamp in Unix format with clock skew correction.
        """
        return self._clock_skew.get_unix_timestamp()

    @staticmethod
    def parse_rfc2616_date(date: str) -> Optional[float]:
        """
        Parses an RFC 2616 date string into a Unix timestamp.

        This function parses an RFC 2616 date string into a Unix timestamp.

        Args:
            date (str): RFC 2616 date string to parse.

        Returns:
            Optional[float]: Unix timestamp of the parsed date string, or None if parsing failed.
        """
        try:
            return (
                dt.strptime(date, "%a, %d %b %Y %H:%M:%S %Z")
                .replace(tzinfo=tz.utc)
                .timestamp()
            )
        except ValueError:
            return None

    def handle_client_response_error(self, e: aiohttp.ClientResponseError) -> None:
        """
        Handle a client response error.

        This method adjusts the clock skew based on the server date in the response headers
        and raises a SkewAdjustmentError if the server date is missing or invalid.

        Args:
            e (Exception): The client response error to handle.

        Returns:
            None
        """
        if e.headers is None:
            raise SkewAdjustmentError("No server date in headers.") from e
        server_date: Optional[str] = e.headers.get("Date", None)
        if server_date is None or not isinstance(server_date, str):
            raise SkewAdjustmentError("No server date in headers.") from e
        server_date_parsed: Optional[float] = DRM.parse_rfc2616_date(server_date)
        if server_date_parsed is None or not isinstance(server_date_parsed, float):
            raise SkewAdjustmentError(
                f"Failed to parse server date: {server_date}"
            ) from e
        client_date = self.get_unix_timestamp()
        self.adj_clock_skew_seconds(server_date_parsed - client_date)

    def generate_sec_ms_gec(self) -> str:
        """
        Generates the Sec-MS-GEC token value.

        This function generates a token value based on the current time in Windows file time format
        adjusted for clock skew, and rounded down to the nearest 5 minutes. The token is then hashed
        using SHA256 and returned as an uppercased hex digest.

        Returns:
            str: The generated Sec-MS-GEC token value.

        See Also:
            https://github.com/rany2/edge-tts/issues/290#issuecomment-2464956570
        """

        # Get the current timestamp in Unix format with clock skew correction
        ticks = self.get_unix_timestamp()

        # Switch to Windows file time epoch (1601-01-01 00:00:00 UTC)
        ticks += WIN_EPOCH

        # Round down to the nearest 5 minutes (300 seconds)
        ticks -= ticks % 300

        # Convert the ticks to 100-nanosecond intervals (Windows file time format)
        ticks *= S_TO_NS / 100

        # Create the string to hash by concatenating the ticks and the trusted client token
        str_to_hash = f"{ticks:.0f}{TRUSTED_CLIENT_TOKEN}"

        # Compute the SHA256 hash and return the uppercased hex digest
        return hashlib.sha256(str_to_hash.encode("ascii")).hexdigest().upper()

    @staticmethod
    def generate_muid() -> str:
        """
        Generates a random MUID.

        Returns:
            str: The generated MUID.
        """
        return secrets.token_hex(16).upper()

    @staticmethod
    def headers_with_muid(headers: Dict[str, str], muid: Optional[str] = None) -> Dict[str, str]:
        """
        Returns a copy of the given headers with the MUID header added.

        Args:
            headers (dict): The original headers.
            muid (str, optional): A pre-determined MUID to use instead of
                generating a random one. When provided, this avoids the
                per-call ``DRM.generate_muid`` mutation that causes
                race conditions under concurrent access.

        Returns:
            dict: The headers with the MUID header added.
        """
        combined_headers = headers.copy()
        assert "Cookie" not in combined_headers
        muid_value = muid if muid is not None else DRM.generate_muid()
        combined_headers["Cookie"] = f"muid={muid_value};"
        return combined_headers


# Shared default DRM instance (uses the shared DEFAULT_CLOCK_SKEW).
DEFAULT_DRM = DRM()
