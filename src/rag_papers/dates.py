"""arXiv announcement calendar.

Submissions close at 14:00 US Eastern on weekdays. Everything received before a
cutoff is announced that evening (Sunday evening for the Friday cutoff) and shows
up in the listing dated the next weekday. The listing date is what this project
calls the "digest day": always a Monday to Friday.

Holidays are not modelled: on those days arXiv skips an announcement and the
papers simply move to the next listing.
"""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
CUTOFF = time(14, 0)
ANNOUNCEMENT = time(20, 0)


def _next_weekday(day: date) -> date:
    day += timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def listing_date(submitted: datetime) -> date:
    """Listing (digest) date for a paper whose v1 was submitted at `submitted`."""
    local = submitted.astimezone(ET)
    cutoff_day = local.date()
    if local.weekday() >= 5 or local.time() >= CUTOFF:
        cutoff_day = _next_weekday(cutoff_day)
    return _next_weekday(cutoff_day)


def latest_listing_date(now: datetime) -> date:
    """The most recent listing already announced at `now`.

    Announcements go out Sunday to Thursday at 20:00 ET, each one for the
    listing dated the following day.
    """
    local = now.astimezone(ET)
    day = local.date()
    if local.time() < ANNOUNCEMENT:
        day -= timedelta(days=1)
    while day.weekday() in (4, 5):  # no announcement on Friday or Saturday evening
        day -= timedelta(days=1)
    return day + timedelta(days=1)


def recent_listing_dates(now: datetime, n: int = 3) -> list[date]:
    """The `n` most recent listings already announced at `now`, newest first."""
    days = [latest_listing_date(now)]
    while len(days) < n:
        day = days[-1] - timedelta(days=1)
        while day.weekday() >= 5:
            day -= timedelta(days=1)
        days.append(day)
    return days
