# models/rule_based.py
import re
from typing import Optional, Sequence, List

CATEGORIES = [
    "Income", "Fixed Expenses", "Variable Expenses",
    "Discretionary", "Transfers", "Loan Repayments", "Bounces",
]

# LAYER 1 — Structural rules
def classify_structural(description: str, txn_type: Optional[str] = None,
                        debit: float = 0, credit: float = 0) -> Optional[str]:
    if not description or not isinstance(description, str):
        return None
    d = description.strip().lower()
    t = (txn_type or "").strip().lower()
    try:
        credit = float(credit or 0)
        debit = float(debit or 0)
    except (TypeError, ValueError):
        credit, debit = 0, 0

    if t == "upi credit" and "/sale" in d:
        return "Income"
    if t == "cash deposit" and "cash deposit-self" in d:
        return "Income"
    if t == "upi debit" and ("supplierpayment" in d or "raw materials" in d):
        return "Variable Expenses"
    if t == "upi debit" and ("familytransfer" in d or "personal" in d):
        return "Transfers"
    if t == "atm withdrawal":
        return "Transfers"
    if t == "debit" and ("rent" in d or "municipal" in d or "maintenance" in d):
        return "Fixed Expenses"

    # Fallback using sign only, when type column is missing/unrecognized
    if credit > 0 and debit == 0 and "sale" in d:
        return "Income"

    return None

# LAYER 2 — Generic keyword rules (fallback for free-text / other banks)
_RAW_RULES = [
    (r"bounce|failed debit|return.*chg|dishonour|insufficient fund|inward return|"
     r"debit.*fail|mandate.*fail|nach.*fail|ecs.*return|chargeback",
     "Bounces"),

    (r"\bemi\b|loan.*repay|repay.*loan|credit.?card.*pay|cc.*pay|"
     r"home.?loan|auto.?loan|personal.?loan|mortgage|lic.*premium|"
     r"bajaj.*fin|hdfc.*loan|sbi.*loan|icici.*loan|axis.*loan",
     "Loan Repayments"),

    (r"salary|sal credit|payroll|neft.*credit|credited.*salary|"
     r"freelance|project.*payment|invoice.*paid|stipend|"
     r"dividend|interest.*credit|fd.*interest|rd.*interest|"
     r"refund|cashback|reversal.*credit|reimbursement|"
     r"sale\b",
     "Income"),

    (r"rent|maintenance|society|municipal|electricity|power.*bill|bescom|"
     r"bses|tata.*power|water.*bill|gas.*bill|igl|mahanagar.*gas|"
     r"broadband|internet.*bill|airtel.*postpaid|jio.*postpaid|vi.*postpaid|"
     r"insurance|lic.*prem|health.*insur|term.*plan|vehicle.*insur",
     "Fixed Expenses"),

    (r"neft.*transfer|imps.*transfer|self.*transfer|own.*account|"
     r"upi.*p2p|transfer.*to|sent.*to|familytransfer|personal|"
     r"atm.*withdrawal|cash.*withdrawal",
     "Transfers"),

    (r"flight|air.*ticket|makemytrip|goibibo|irctc|train.*ticket|"
     r"hotel|booking\.com|airbnb|oyo|"
     r"amazon\.in|flipkart|myntra|ajio|nykaa|meesho|"
     r"gold|jewel|iphone|macbook|gadget|luxury|premium.*subscription|"
     r"netflix|spotify|prime.*video|hotstar|zee5|"
     r"gaming|steam|playstation|xbox",
     "Discretionary"),

    (r"swiggy|zomato|dunzo|blinkit|zepto|instamart|"
     r"grocery|vegetables|fruits|kirana|bigbasket|"
     r"supplier|raw material|"
     r"uber|ola|rapido|metro|autorickshaw|"
     r"pharmacy|medical|chemist|hospital|clinic|doctor|"
     r"petrol|fuel|hp.*petrol|indian.*oil|bharat.*petrol|"
     r"restaurant|cafe|coffee|food|dining|dominos|mcdonalds|kfc|"
     r"milk|dairy|bakery",
     "Variable Expenses"),
]

_RULES = [
    (re.compile(pattern, re.IGNORECASE), category)
    for pattern, category in _RAW_RULES
]

def classify(description: str, txn_type: Optional[str] = None,
             debit: float = 0, credit: float = 0) -> Optional[str]:
    # Layer 1: structural (type + credit/debit aware)
    struct = classify_structural(description, txn_type, debit, credit)
    if struct:
        return struct
    # Layer 2: generic keyword fallback
    if not description or not isinstance(description, str):
        return None
    for pattern, category in _RULES:
        if pattern.search(description.strip()):
            return category
    return None

def batch_classify(descriptions: Sequence[str]) -> List[Optional[str]]:
    return [classify(d) for d in descriptions]