import unittest
from datetime import date

from resume_pipeline import (
    fallback_summary,
    normalize_resume_facts,
    summary_is_acceptable,
    summary_source_payload,
)


class ResumePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.today = date(2026, 8, 1)

    def test_recovers_range_and_calculates_nonzero_experience(self) -> None:
        raw = {
            "roles": [
                {
                    "title": "Product Manager",
                    "organization": "Example Health",
                    "location": "",
                    "date_text": "2022-present",
                    "start_date_text": "2022",
                    "end_date_text": "",
                    "is_current": True,
                    "highlights": [
                        "Leads product strategy for patient engagement software."
                    ],
                },
                {
                    "title": "Analyst",
                    "organization": "Example Consulting",
                    "location": "",
                    "date_text": "2019 to 2022",
                    "start_date_text": "2019",
                    "end_date_text": "",
                    "is_current": False,
                    "highlights": [],
                },
            ],
            "education": [],
            "industries": ["Healthcare technology"],
            "professional_functions": ["Product strategy"],
            "skills": ["SQL", "sql"],
            "career_topics": [],
            "evidence_notes": [],
        }

        result = normalize_resume_facts(raw, today=self.today)

        self.assertEqual(result["years_experience_estimate"], 7.1)
        self.assertEqual(result["work_history"][1]["end_date"], "2022")
        self.assertEqual(result["skills"], ["SQL"])
        self.assertIsNone(result["work_history"][0]["location"])

    def test_overlapping_roles_are_not_double_counted(self) -> None:
        raw = {
            "roles": [
                {
                    "title": "Role A",
                    "organization": "A",
                    "location": "",
                    "date_text": "2020 to 2024",
                    "start_date_text": "2020",
                    "end_date_text": "2024",
                    "is_current": False,
                    "highlights": [],
                },
                {
                    "title": "Role B",
                    "organization": "B",
                    "location": "",
                    "date_text": "2022 to 2023",
                    "start_date_text": "2022",
                    "end_date_text": "2023",
                    "is_current": False,
                    "highlights": [],
                },
            ]
        }

        result = normalize_resume_facts(raw, today=self.today)
        self.assertEqual(result["years_experience_estimate"], 4.0)

    def test_rejects_corrupted_summary(self) -> None:
        profile = normalize_resume_facts(
            {
                "roles": [
                    {
                        "title": "Analyst",
                        "organization": "Example Consulting",
                        "location": "",
                        "date_text": "2019 to 2022",
                        "start_date_text": "2019",
                        "end_date_text": "2022",
                        "is_current": False,
                        "highlights": [],
                    }
                ]
            },
            today=self.today,
        )
        payload = summary_source_payload(profile)

        self.assertFalse(
            summary_is_acceptable(
                "Previously served as an analyst from 2019 to 2 as a graduate.",
                source_payload=payload,
            )
        )

    def test_fallback_summary_uses_validated_facts(self) -> None:
        profile = {
            "current_or_most_recent_role": {
                "title": "Product Manager",
                "organization": "Example Health",
            },
            "work_history": [
                {
                    "title": "Product Manager",
                    "organization": "Example Health",
                },
                {
                    "title": "Analyst",
                    "organization": "Example Consulting",
                },
            ],
            "education": [
                {
                    "degree": "B.A. in Economics",
                    "institution": "Example University",
                }
            ],
            "skills": ["SQL"],
        }

        self.assertEqual(
            fallback_summary(profile),
            "Works as Product Manager at Example Health. Previously worked as Analyst at Example Consulting. Holds a B.A. in Economics from Example University.",
        )


if __name__ == "__main__":
    unittest.main()
