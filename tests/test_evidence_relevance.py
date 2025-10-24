import pytest

from arbee.tools.evidence import is_relevant_to_subclaim


@pytest.mark.parametrize(
    "title,content",
    [
        (
            "Global weather trends could reshape marathons",
            "Meteorologists warn that hotter races will challenge runners worldwide.",
        ),
        (
            "Climate change is making endurance events tougher",
            "Experts discuss how humidity and heat impact marathon performance.",
        ),
    ],
)
def test_relevance_requires_subject_when_subclaim_mentions_subject(title: str, content: str) -> None:
    subclaim = "Diplo has been training specifically for running events in 2025, improving his chances."
    market_question = "Will Diplo run a 5k in under 23 minutes in 2025?"

    assert not is_relevant_to_subclaim(title, content, subclaim, market_question)


def test_relevance_accepts_subject_alias_mentions() -> None:
    title = "Thomas Wesley Pentz adds new races to his 2025 run club schedule"
    content = (
        "Thomas Wesley Pentz, better known as Diplo, announced additional 5k events "
        "and described how his training volume has increased this autumn."
    )
    subclaim = "Diplo has been training specifically for running events in 2025, improving his chances."
    market_question = "Will Diplo run a 5k in under 23 minutes in 2025?"

    assert is_relevant_to_subclaim(title, content, subclaim, market_question)


def test_relevance_allows_general_subclaims_without_subject_requirement() -> None:
    title = "Average 5k finishing times for recreational runners improved in 2025"
    content = (
        "Data from major road races shows recreational runners aged 40-45 average "
        "a 5k finish time of around 26 minutes."
    )
    subclaim = "Recreational runners aged 40 to 45 tend to run 5k races slower than 23 minutes."
    market_question = "Will Diplo run a 5k in under 23 minutes in 2025?"

    assert is_relevant_to_subclaim(title, content, subclaim, market_question)
