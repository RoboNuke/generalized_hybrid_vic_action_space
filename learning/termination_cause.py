"""Canonical episode termination-cause codes — the SINGLE source of truth.

Every finished episode has exactly one *cause*. The env stamps a per-env integer code onto
``env._termination_cause`` at ``_get_dones`` time (``NaN`` for envs that did not finish this step)
and publishes it in the eval step-trace as the ``termination_cause`` column. Downstream readers
(the eval-analysis notebook) map codes -> names via :data:`CAUSE_NAMES` and choose which causes to
count toward a given plot (e.g. break rate = fraction of episodes whose cause is ``"peg_break"``).

Add a new cause by appending a ``(code, name)`` pair below with the next unused positive int, and
have whatever decides that termination stamp the matching constant. Rules that keep old data valid:

* **Never renumber or reuse a code.** Old traces carry the old integers; renumbering silently
  relabels history.
* Old traces simply never carry a newly-added code, and a reader that meets an unknown code falls
  back to ``str(code)`` — so nothing breaks when causes are added.

This module has NO heavy imports on purpose, so both the Isaac env and a plain analysis notebook
can import it.
"""

# code -> human name. Code 0 is reserved for "did not finish this step" (the trace stores NaN for
# that, but 0 is kept free so a real cause code is never falsy). Codes are STABLE forever.
CAUSE_NAMES = {
    1: "success",       # reached the goal pose (with the keypoint-coverage gate) while in contact
    2: "timeout",       # ran out of episode steps without traversing the whole path
    3: "traversed",     # progress frontier passed every keypoint (path fully covered) -> truncated
    4: "lag",           # fell too far behind the moving setpoint while in contact
    5: "peg_break",     # fragile peg: contact force exceeded the break threshold
    6: "contact_loss",  # fragile require-contact: dropped out of contact after first contact
}
NAME_TO_CODE = {name: code for code, name in CAUSE_NAMES.items()}


def cause_name(code) -> str:
    """Map a (possibly float / unknown) cause code to its name, falling back to ``str(int(code))``."""
    try:
        return CAUSE_NAMES.get(int(code), str(int(code)))
    except (TypeError, ValueError):
        return str(code)


# Named constants for the producers (env / wrappers) so they never hardcode the ints.
SUCCESS = 1
TIMEOUT = 2
TRAVERSED = 3
LAG = 4
PEG_BREAK = 5
CONTACT_LOSS = 6
