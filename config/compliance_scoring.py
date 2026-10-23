"""Central, versioned policy for the audit compliance score.

Weights are relative category weights. Checks that do not apply to an audit are
excluded, then the remaining weights are normalized to the available audit.
Unresolved checks remain in the denominator with zero earned points and reduce
coverage; this is deliberately conservative for incomplete evidence.
"""

SCORING_VERSION = "1.1.0"
SCORING_WEIGHTS = {
    "placement": 0.80,
    "price": 0.20,
}
