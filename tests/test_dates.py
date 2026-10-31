from datetime import date, datetime

import pytest

from rag_papers.dates import ET, latest_listing_date, recent_listing_dates, listing_date


def et(*args):
    return datetime(*args, tzinfo=ET)


# Week of Mon 2024-10-07 to Sun 2024-10-13.
@pytest.mark.parametrize(
    "submitted, expected",
    [
        (et(2024, 10, 7, 10, 0), date(2024, 10, 8)),  # Mon before cutoff -> Tue
        (et(2024, 10, 7, 14, 0), date(2024, 10, 9)),  # Mon at cutoff -> Wed
        (et(2024, 10, 10, 13, 59), date(2024, 10, 11)),  # Thu before cutoff -> Fri
        (et(2024, 10, 10, 15, 0), date(2024, 10, 14)),  # Thu after cutoff -> Mon
        (et(2024, 10, 11, 13, 0), date(2024, 10, 14)),  # Fri before cutoff -> Mon
        (et(2024, 10, 11, 15, 0), date(2024, 10, 15)),  # Fri after cutoff -> Tue
        (et(2024, 10, 12, 9, 0), date(2024, 10, 15)),  # Saturday -> Tue
        (et(2024, 10, 13, 22, 0), date(2024, 10, 15)),  # Sunday -> Tue
    ],
)
def test_listing_date(submitted, expected):
    assert listing_date(submitted) == expected


def test_listing_date_converts_from_utc():
    # 2024-10-07 19:30 UTC is 15:30 EDT, after the Monday cutoff.
    assert listing_date(datetime.fromisoformat("2024-10-07T19:30:00+00:00")) == date(2024, 10, 9)


def test_listing_date_across_dst_change():
    # DST ends 2024-11-03. Mon 2024-11-04 18:30 UTC is 13:30 EST, before the cutoff.
    assert listing_date(datetime.fromisoformat("2024-11-04T18:30:00+00:00")) == date(2024, 11, 5)


@pytest.mark.parametrize(
    "now, expected",
    [
        (et(2024, 10, 7, 19, 0), date(2024, 10, 7)),  # Mon before announcement -> Mon listing
        (et(2024, 10, 7, 20, 30), date(2024, 10, 8)),  # Mon after announcement -> Tue
        (et(2024, 10, 11, 23, 0), date(2024, 10, 11)),  # Fri night: still Fri listing
        (et(2024, 10, 12, 12, 0), date(2024, 10, 11)),  # Sat -> Fri listing
        (et(2024, 10, 13, 19, 0), date(2024, 10, 11)),  # Sun before announcement -> Fri
        (et(2024, 10, 13, 20, 0), date(2024, 10, 14)),  # Sun announcement -> Mon
    ],
)
def test_latest_listing_date(now, expected):
    assert latest_listing_date(now) == expected


def test_recent_listing_dates_skip_weekends():
    # Tuesday morning: Tue listing was announced Monday evening.
    assert recent_listing_dates(et(2024, 10, 8, 3, 0)) == [date(2024, 10, 8), date(2024, 10, 7), date(2024, 10, 4)]
