"""
DATASET CONFIGURATION MODULE (NATURAL LANGUAGE EDITION)
-------------------------------------------------------
This module defines controlled natural-language narratives as experimental stimuli.
Each narrative is paired with a Z3-based solver function that provides the ground-truth label.

Label space (consistent across the dataset):
- True    : entailed / necessarily true given the premises
- False   : contradicted / necessarily false given the premises
- Unknown : underdetermined (both truth values are satisfiable under the premises)
- Paradox : inconsistent premises (unsatisfiable knowledge base)
"""

from solvers import *

ALL_PUZZLES = [
    {
        "id": "P01_Chain_Factory",
        "category": "Transitive",
        "text": (
            "Story: In a robot factory, there is a strict chain of events.\n"
            "1. If the Red Button is pressed, the Main Gear turns.\n"
            "2. If the Main Gear turns, the Whistle blows.\n"
            "3. If the Whistle blows, the Packing Box closes.\n"
            "Observation: The Packing Box is NOT closed.\n"
            "Question: Was the Red Button pressed?"
        ),
        "z3_func": z3_p1,
    },
    {
        "id": "P02_Contradiction_Potion",
        "category": "Transitive",
        "text": (
            "Story: An alchemist created a potion with specific effects.\n"
            "1. If Alice drinks the potion, Alice becomes Invisible.\n"
            "2. If Alice drinks the potion, Alice becomes Strong.\n"
            "Observation: Alice is Invisible, but Alice is NOT Strong.\n"
            "Question: Did Alice drink the potion?"
        ),
        "z3_func": z3_p2,
    },
    {
        "id": "P03_Exclusion_Theft",
        "category": "Sets",
        "text": (
            "Story: A diamond was stolen. Exactly one person is guilty: the Butler, the Gardener, or the Chef.\n"
            "1. If the Butler is guilty, there are muddy footprints.\n"
            "2. If the Gardener is guilty, there are NO muddy footprints.\n"
            "3. If the Chef is guilty, there are NO muddy footprints.\n"
            "Observation: Muddy footprints were found.\n"
            "Question: Is the Butler guilty?"
        ),
        "z3_func": z3_p3,
    },
    {
        "id": "P04_Counting_PowerPlant",
        "category": "Sets",
        "text": (
            "Story: A power plant has three generators: G1, G2, and G3.\n"
            "Constraint: Exactly two generators must be active at the same time.\n"
            "Observation: Generator G1 is OFF.\n"
            "Question: Are both G2 and G3 active?"
        ),
        "z3_func": z3_p4,
    },
    {
    "id": "P05_Satisfiability_TimeTravel",
    "category": "Meta-Logic",
        "text": (
            "Story: Consider the following time-travel rules.\n"
            "1. If I travel back in time, I meet my grandfather.\n"
            "2. If I meet my grandfather, I distract him from meeting my grandmother.\n"
            "3. If I distract him, I am never born.\n"
            "4. If I am never born, I cannot travel back in time.\n"
            "Question: Is the set of facts logically satisfiable (i.e., is there a consistent scenario)?"
        ),
        # IMPORTANT: z3_p5 must return "True" if SAT, otherwise "Paradox"
        "z3_func": z3_p5,
    "valid_outputs": ["True", "Paradox"],
},

    
    {
        "id": "P06_Exception_Penguin",
        "category": "Non-Monotonic",
        "text": (
            "Story: We use a default rule about birds.\n"
            "1. Normally, if an animal is a Bird, it can Fly.\n"
            "2. Being a Penguin is an exception to this rule.\n"
            "Observation: Tweety is a Bird. Tweety is also a Penguin.\n"
            "Question: Can Tweety Fly?"
        ),
        "z3_func": z3_p6,
    },
    {
        "id": "P07_Independence_Rooms",
        "category": "Fallacy",
        "text": (
            "Story: There are two independent rooms.\n"
            "1. In Room A: If Switch A is flipped, the Fan turns on.\n"
            "2. In Room B: If Switch B is flipped, the Light turns on.\n"
            "Observation: In Room B, Switch B was flipped and the Light is on.\n"
            "Question: Is the Fan in Room A on?"
        ),
        "z3_func": z3_p7,
    },
    {
        "id": "P08_Tautology_StockMarket",
        "category": "Propositional",
        "text": (
            "Story: A trader considers a stock.\n"
            "1. If the market goes UP, the trader makes a profit.\n"
            "2. If the market does NOT go UP, the trader still makes a profit.\n"
            "Question: Will the trader make a profit?"
        ),
        "z3_func": z3_p8,
    },
    {
        "id": "P09_Paradox_ContradictoryObservation",
        "category": "Paradox",
        "text": (
            "Story: A report contains two statements about the same lamp.\n"
            "1. The lamp is ON.\n"
            "2. The lamp is NOT ON.\n"
            "Question: Is the lamp ON?"
        ),
        "z3_func": z3_p9,
    },
]

if __name__ == "__main__":
    print("[SYSTEM] Narrative dataset loaded.")
    sample = ALL_PUZZLES[0]
    print(f"[TEST] {sample['id']} => {sample['z3_func']()}")
