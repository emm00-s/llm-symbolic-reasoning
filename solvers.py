from z3 import *

# FORMAL VERIFICATION ENGINE (Helper Function)
def check_full_truth(solver, target_var):
    """
    Determines the strict logical necessity of a target variable.
    
    Returns:
    - "True": If target is entailed by premises (Necessarily True).
    - "False": If negation of target is entailed (Necessarily False).
    - "Unknown": If target is contingent (Satisfiable but not Valid).
    - "Paradox": If premises are inconsistent (Global UNSAT).
    
    Method: Proof by Contradiction (Reductio ad Absurdum).
    """
    # 1. Consistency Check
    if solver.check() == unsat: return "Paradox"
    
    # 2. Check Positive Necessity (Does Not(Target) lead to contradiction?)
    solver.push()
    solver.add(Not(target_var))
    must_be_true = (solver.check() == unsat)
    solver.pop()
    if must_be_true: return "True"

    # 3. Check Negative Necessity (Does Target lead to contradiction?)
    solver.push()
    solver.add(target_var)
    must_be_false = (solver.check() == unsat)
    solver.pop()
    if must_be_false: return "False"

    # 4. Contingency
    return "Unknown"

# PUZZLE FORMALIZATION (Z3 Constraints)

def z3_p1():
    # P1: Transitive Inference & Modus Tollens.
    # Given chain A->B->C->D and not(D), infers not(A).
    s = Solver()
    A, B, C, D = Bools('A B C D')
    s.add(Implies(A, B), Implies(B, C), Implies(C, D), Not(D))
    return check_full_truth(s, A) 

def z3_p2():
    # P2: Direct Contradiction.
    # Premises A->B, A->C, B, and not(C) imply not(A).
    s = Solver()
    A, B, C = Bools('A B C')
    s.add(Implies(A, B), Implies(A, C), B, Not(C))
    return check_full_truth(s, A)

def z3_p3():
    # P3: Exclusive Disjunction (XOR) & Implication.
    # Constraints include a Pseudo-Boolean equality (Sum=1).
    # Fixed: Added Implies(C, Not(D)) so Chef also leaves no footprints.
    s = Solver()
    A, B, C, D = Bools('A B C D')
    s.add(PbEq([(A,1), (B,1), (C,1)], 1)) # Exactly One constraint
    s.add(Implies(A, D), Implies(B, Not(D)), Implies(C, Not(D)), D)
    return check_full_truth(s, A)

def z3_p4():
    # P4: Cardinality Constraints.
    # Deduces states of B and C given Exactly-Two constraint and not(A).
    s = Solver()
    A, B, C = Bools('A B C')
    s.add(PbEq([(A,1), (B,1), (C,1)], 2), Not(A))
    return check_full_truth(s, And(B, C))

def z3_p5():
    # P5: META-LOGIC — Satisfiability Check (Grandfather Paradox)
    #
    # Domanda diversa da tutti gli altri puzzle: non testa il valore di una
    # variabile target, ma la consistenza globale del knowledge base.
    #
    # Output space ridotto: {"True" (SAT), "Paradox" (UNSAT)}.
    # "False" e "Unknown" non sono possibili in questo contesto.
    #
    # Nota: la catena T->M->D->Not(B)->Not(T) è un'implicazione circolare,
    # NON una contraddizione. Es. modello valido: T=False, B=True.
    # Il LLM potrebbe rispondere "Paradox" per bias narrativo sul Grandfather
    # Paradox — questa è la trappola semantica del puzzle.
    s = Solver()
    T, M, D, B = Bools('T M D B')  # Travel, Meet, Distract, Born
    s.add(Implies(T, M))
    s.add(Implies(M, D))
    s.add(Implies(D, Not(B)))
    s.add(Implies(Not(B), Not(T)))
    return "True" if s.check() == sat else "Paradox"

def z3_p6():
    # P6: Defeasible Reasoning (Non-Monotonic Logic).
    # Models an exception (C) that overrides a default rule (A->B).
    s = Solver()
    A, B, C = Bools('A B C')
    s.add(Implies(And(A, Not(C)), B), A, C)
    return check_full_truth(s, B)

def z3_p7():
    # P7: Logical Independence.
    # Tests if truth of B is independent of premises regarding C and D.
    s = Solver()
    A, B, C, D = Bools('A B C D')
    s.add(Implies(A, B), Implies(C, D), C, D)
    return check_full_truth(s, B)

def z3_p8():
    # P8: Law of Excluded Middle (Proof by Cases).
    # If A->B and not(A)->B, then B is valid regardless of A.
    s = Solver()
    A, B = Bools('A B')
    s.add(Implies(A, B), Implies(Not(A), B))
    return check_full_truth(s, B)

def z3_p9():
    # P9: Contradiction (Paradox).
    # Premesse L e Not(L) sono inconsistenti. Il consistency check in
    # check_full_truth restituisce "Paradox" prima ancora di testare il target.
    s = Solver()
    L = Bool('L')
    s.add(L)
    s.add(Not(L))
    return check_full_truth(s, L)