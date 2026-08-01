from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable

MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

PRESENT_TERMS = {
    "present",
    "current",
    "currently",
    "now",
    "ongoing",
    "today",
}

DATE_TOKEN = (
    r"(?:"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}"
    r"|\d{1,2}[/-]\d{4}"
    r"|\d{4}[/-]\d{1,2}"
    r"|\d{4}"
    r"|present|current|currently|now|ongoing|today"
    r")"
)

RANGE_PATTERN = re.compile(
    rf"(?P<start>{DATE_TOKEN})\s*(?:-|–|—|to|through|until)\s*(?P<end>{DATE_TOKEN})",
    re.IGNORECASE,
)

YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")


@dataclass(frozen=True)
class ParsedDate:
    value: date
    precision: str
    original: str
    is_present: bool = False


def clean_optional_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.strip().split())
    return cleaned or None


def clean_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []

    seen: set[str] = set()
    cleaned_values: list[str] = []

    for value in values:
        cleaned = clean_optional_string(value)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned_values.append(cleaned)

    return cleaned_values


def parse_resume_date(
    value: str | None,
    *,
    today: date,
) -> ParsedDate | None:
    cleaned = clean_optional_string(value)
    if not cleaned:
        return None

    normalized = cleaned.casefold().strip(".,;:()[]{}")
    if normalized in PRESENT_TERMS:
        return ParsedDate(
            value=today,
            precision="present",
            original=cleaned,
            is_present=True,
        )

    full_date_match = re.fullmatch(
        r"(\d{4})-(\d{1,2})-(\d{1,2})",
        normalized,
    )
    if full_date_match:
        year, month, day = map(int, full_date_match.groups())
        try:
            return ParsedDate(
                value=date(year, month, day),
                precision="day",
                original=cleaned,
            )
        except ValueError:
            return None

    year_month_match = re.fullmatch(
        r"(\d{4})[/-](\d{1,2})",
        normalized,
    )
    if year_month_match:
        year, month = map(int, year_month_match.groups())
        if 1 <= month <= 12:
            return ParsedDate(
                value=date(year, month, 15),
                precision="month",
                original=cleaned,
            )
        return None

    month_year_numeric_match = re.fullmatch(
        r"(\d{1,2})[/-](\d{4})",
        normalized,
    )
    if month_year_numeric_match:
        month, year = map(int, month_year_numeric_match.groups())
        if 1 <= month <= 12:
            return ParsedDate(
                value=date(year, month, 15),
                precision="month",
                original=cleaned,
            )
        return None

    month_year_match = re.fullmatch(
        r"([a-z]+)\s+(\d{4})",
        normalized,
    )
    if month_year_match:
        month_text, year_text = month_year_match.groups()
        month = MONTHS.get(month_text)
        if month:
            return ParsedDate(
                value=date(int(year_text), month, 15),
                precision="month",
                original=cleaned,
            )
        return None

    year_match = re.fullmatch(r"((?:19|20)\d{2})", normalized)
    if year_match:
        # A midpoint convention avoids systematically treating a bare year
        # as either January 1 or December 31.
        return ParsedDate(
            value=date(int(year_match.group(1)), 7, 1),
            precision="year",
            original=cleaned,
        )

    return None


def recover_date_range(date_text: str | None) -> tuple[str | None, str | None]:
    cleaned = clean_optional_string(date_text)
    if not cleaned:
        return None, None

    match = RANGE_PATTERN.search(cleaned)
    if not match:
        return None, None

    return (
        clean_optional_string(match.group("start")),
        clean_optional_string(match.group("end")),
    )


def role_sort_value(role: dict[str, Any]) -> tuple[int, date, date]:
    end = role.get("_end_date_value")
    start = role.get("_start_date_value")
    return (
        1 if role.get("is_current") else 0,
        end if isinstance(end, date) else date.min,
        start if isinstance(start, date) else date.min,
    )


def public_role(role: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": role.get("title"),
        "organization": role.get("organization"),
        "location": role.get("location"),
        "date_text": role.get("date_text"),
        "start_date": role.get("start_date"),
        "end_date": role.get("end_date"),
        "start_date_iso": role.get("start_date_iso"),
        "end_date_iso": role.get("end_date_iso"),
        "is_current": bool(role.get("is_current")),
        "highlights": role.get("highlights", []),
    }


def normalize_role(
    raw_role: Any,
    *,
    today: date,
    warnings: list[str],
) -> dict[str, Any] | None:
    if not isinstance(raw_role, dict):
        return None

    title = clean_optional_string(raw_role.get("title"))
    organization = clean_optional_string(raw_role.get("organization"))
    location = clean_optional_string(raw_role.get("location"))
    date_text = clean_optional_string(raw_role.get("date_text"))
    start_text = clean_optional_string(raw_role.get("start_date_text"))
    end_text = clean_optional_string(raw_role.get("end_date_text"))
    is_current = bool(raw_role.get("is_current"))
    highlights = clean_string_list(raw_role.get("highlights"))

    recovered_start, recovered_end = recover_date_range(date_text)
    if not start_text:
        start_text = recovered_start
    if not end_text:
        end_text = recovered_end

    start = parse_resume_date(start_text, today=today)
    end = parse_resume_date(end_text, today=today)

    if end and end.is_present:
        is_current = True

    if is_current and end is None:
        end = ParsedDate(
            value=today,
            precision="present",
            original="Present",
            is_present=True,
        )
        end_text = "Present"

    if start and end and end.value < start.value:
        warnings.append(
            "A role had an end date earlier than its start date and was "
            "excluded from the experience calculation."
        )

    if not any((title, organization, date_text, start_text, end_text, highlights)):
        return None

    return {
        "title": title,
        "organization": organization,
        "location": location,
        "date_text": date_text,
        "start_date": start.original if start else start_text,
        "end_date": (
            "Present"
            if is_current
            else (end.original if end else end_text)
        ),
        "start_date_iso": start.value.isoformat() if start else None,
        "end_date_iso": end.value.isoformat() if end else None,
        "is_current": is_current,
        "highlights": highlights,
        "_start_date_value": start.value if start else None,
        "_end_date_value": end.value if end else None,
        "_start_precision": start.precision if start else None,
        "_end_precision": end.precision if end else None,
    }


def merge_intervals(intervals: Iterable[tuple[date, date]]) -> list[tuple[date, date]]:
    sorted_intervals = sorted(intervals, key=lambda item: item[0])
    merged: list[tuple[date, date]] = []

    for start, end in sorted_intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue

        previous_start, previous_end = merged[-1]
        if end > previous_end:
            merged[-1] = (previous_start, end)

    return merged


def calculate_experience_years(
    roles: list[dict[str, Any]],
    *,
    warnings: list[str],
) -> float | None:
    intervals: list[tuple[date, date]] = []
    used_year_precision = False

    for role in roles:
        start = role.get("_start_date_value")
        end = role.get("_end_date_value")

        if not isinstance(start, date) or not isinstance(end, date):
            if isinstance(start, date) and not role.get("is_current"):
                warnings.append(
                    "A role with a start date but no end date was excluded "
                    "from the experience estimate."
                )
            continue

        if end < start:
            continue

        if (
            role.get("_start_precision") == "year"
            or role.get("_end_precision") == "year"
        ):
            used_year_precision = True

        intervals.append((start, end))

    if not intervals:
        return None

    if used_year_precision:
        warnings.append(
            "Year-only employment dates were converted to July 1 for the "
            "experience estimate."
        )

    merged = merge_intervals(intervals)
    total_days = sum((end - start).days for start, end in merged)
    return round(total_days / 365.2425, 1)


def normalize_education(raw_education: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_education, list):
        return []

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for item in raw_education:
        if not isinstance(item, dict):
            continue

        record = {
            "institution": clean_optional_string(item.get("institution")),
            "degree": clean_optional_string(item.get("degree")),
            "field": clean_optional_string(item.get("field")),
            "graduation_year": clean_optional_string(
                item.get("graduation_year")
            ),
        }

        if not any(record.values()):
            continue

        key = tuple((record[field] or "").casefold() for field in record)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(record)

    return normalized


def normalize_resume_facts(
    raw_facts: dict[str, Any],
    *,
    today: date,
) -> dict[str, Any]:
    warnings: list[str] = []

    raw_roles = raw_facts.get("roles")
    roles: list[dict[str, Any]] = []
    if isinstance(raw_roles, list):
        for raw_role in raw_roles:
            role = normalize_role(
                raw_role,
                today=today,
                warnings=warnings,
            )
            if role:
                roles.append(role)

    roles.sort(key=role_sort_value, reverse=True)
    years_experience = calculate_experience_years(
        roles,
        warnings=warnings,
    )

    current_or_recent = public_role(roles[0]) if roles else None
    public_roles = [public_role(role) for role in roles]

    evidence_notes = clean_string_list(raw_facts.get("evidence_notes"))
    evidence_notes.extend(warnings)
    evidence_notes = clean_string_list(evidence_notes)

    return {
        "current_or_most_recent_role": current_or_recent,
        "work_history": public_roles,
        "education": normalize_education(raw_facts.get("education")),
        "industries": clean_string_list(raw_facts.get("industries")),
        "professional_functions": clean_string_list(
            raw_facts.get("professional_functions")
        ),
        "skills": clean_string_list(raw_facts.get("skills")),
        "career_topics": clean_string_list(
            raw_facts.get("career_topics")
        ),
        "years_experience_estimate": years_experience,
        # These should be confirmed by the member rather than inferred from
        # employment history alone.
        "can_advise_on": [],
        "possible_learning_interests": [],
        "evidence_notes": evidence_notes,
        "validation": {
            "date_calculation_method": (
                "Employment intervals are merged to avoid double counting. "
                "Bare years use July 1 as a midpoint."
            ),
            "warnings": warnings,
        },
    }


def summary_source_payload(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_or_most_recent_role": profile.get(
            "current_or_most_recent_role"
        ),
        "work_history": profile.get("work_history", []),
        "education": profile.get("education", []),
        "industries": profile.get("industries", []),
        "professional_functions": profile.get(
            "professional_functions", []
        ),
        "skills": profile.get("skills", []),
        "career_topics": profile.get("career_topics", []),
        "years_experience_estimate": profile.get(
            "years_experience_estimate"
        ),
    }


def summary_is_acceptable(
    summary: Any,
    *,
    source_payload: dict[str, Any],
) -> bool:
    if not isinstance(summary, str):
        return False

    cleaned = " ".join(summary.split())
    if len(cleaned) < 40 or len(cleaned) > 750:
        return False

    # Reject obvious truncated year ranges such as "2019 to 2 as...".
    if re.search(r"\b(?:19|20)\d{2}\s+to\s+\d{1,3}\b", cleaned):
        return False

    allowed_years = set(
        YEAR_PATTERN.findall(
            str(source_payload)
        )
    )
    summary_years = set(YEAR_PATTERN.findall(cleaned))
    if not summary_years.issubset(allowed_years):
        return False

    return True


def fallback_summary(profile: dict[str, Any]) -> str:
    sentences: list[str] = []
    current = profile.get("current_or_most_recent_role")

    if isinstance(current, dict):
        title = clean_optional_string(current.get("title"))
        organization = clean_optional_string(current.get("organization"))
        if title and organization:
            sentences.append(f"Works as {title} at {organization}.")
        elif title:
            sentences.append(f"Works as {title}.")
        elif organization:
            sentences.append(f"Has professional experience at {organization}.")

    prior_roles = profile.get("work_history")
    if isinstance(prior_roles, list):
        for role in prior_roles[1:]:
            if not isinstance(role, dict):
                continue
            title = clean_optional_string(role.get("title"))
            organization = clean_optional_string(role.get("organization"))
            if title and organization:
                sentences.append(
                    f"Previously worked as {title} at {organization}."
                )
                break

    education = profile.get("education")
    if isinstance(education, list) and education:
        first = education[0]
        if isinstance(first, dict):
            degree = clean_optional_string(first.get("degree"))
            institution = clean_optional_string(first.get("institution"))
            if degree and institution:
                sentences.append(f"Holds a {degree} from {institution}.")
            elif institution:
                sentences.append(f"Studied at {institution}.")

    if len(sentences) < 3:
        skills = clean_string_list(profile.get("skills"))[:4]
        if skills:
            sentences.append(f"Skills include {', '.join(skills)}.")

    if not sentences:
        return "Professional background information was extracted from the résumé and is awaiting member review."

    return " ".join(sentences[:3])
