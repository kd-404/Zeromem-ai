# ZeroMem document test: Kaveri Loom (fictional company)

Start ZeroMem on the sample documents (from the `zeromem-ai/` folder, `.venv` active):

```
python -m zeromem.chat_app --docs samples/company_docs                  # full system
python -m zeromem.chat_app --docs samples/company_docs --zeromem-only   # ZeroMem alone
```

All four files are **fictional sample data**. Kaveri Loom is not a real company.

| File | Format | How it is read |
|---|---|---|
| `kaveri_loom_company_handbook.pdf` | PDF, 4 pages | PDF text |
| `scanned_holiday_notice.png` | image (a "scan") | **OCR** |
| `travel_expense_policy.docx` | Word | Word reader |
| `product_price_list.csv` | CSV | one sentence per row |

Score each answer: **right** (the expected sentence), **wrong** (a different sentence), or **refused**.

## A. Should answer: one clear sentence exists

| # | Question | Expected answer (contains) | File |
|---|---|---|---|
| 1 | Who is the CEO of Kaveri Loom? | Meera Raghunathan is the Chief Executive Officer | handbook p.1 |
| 2 | When was the company founded? | founded in 2014 by Meera Raghunathan and Arjun Das | handbook p.1 |
| 3 | How many people work at Kaveri Loom? | employs 1,240 people | handbook p.1 |
| 4 | Where is the design studio? | design studio is in Bengaluru | handbook p.1 |
| 5 | Who is the CFO? | Karthik Iyer joined as Chief Financial Officer | handbook p.1 |
| 6 | What was the total revenue in FY 2025-26? | INR 412 crore | handbook p.2 |
| 7 | What was the net profit? | INR 38.6 crore | handbook p.2 |
| 8 | Which is the largest export market? | Germany was the company's largest export market | handbook p.2 |
| 9 | How many days of paid annual leave do employees get? | 24 days of paid annual leave | handbook p.2 |
| 10 | What is the notice period for managers? | 60 days for managers | handbook p.2 |
| 11 | How long is maternity leave? | Maternity leave is 26 weeks | handbook p.2 |
| 12 | How long must passwords be? | at least 14 characters | handbook p.3 |
| 13 | How quickly must security incidents be reported? | within one hour | handbook p.3 |
| 14 | How long are backups kept? | kept for 90 days | handbook p.3 |
| 15 | Since when is the company ISO 27001 certified? | ISO/IEC 27001 since 2022 | handbook p.3 |
| 16 | What is the return period for customers? | within 30 days of delivery | handbook p.3 |
| 17 | When will the Tiruppur factory be closed? | closed on 2 October 2026 for Gandhi Jayanti | **scan (OCR)** |
| 18 | What are the Diwali holiday dates? | 20 October to 22 October 2026 | **scan (OCR)** |
| 19 | What is the daily meal allowance for travel? | INR 1,500 | **Word** |
| 20 | How soon must expense claims be submitted? | within 15 days of returning | **Word** |
| 21 | How much does the heavyweight hoodie cost? | Heavyweight hoodie ... Price (INR): 1899 | **CSV** |

## B. Should refuse: the documents don't say

| # | Question |
|---|---|
| 22 | Who is the CEO of Tata Motors? |
| 23 | What is the company's share price? |
| 24 | How many stores does Kaveri Loom have in the USA? |
| 25 | What is the canteen menu on Friday? |

## C. Hard on purpose (known weak spots)

| # | Question | Why it's hard |
|---|---|---|
| 26 | What was the revenue in Q3? | the answer is in a **table** (121), not a sentence |
| 27 | What is the address of the Coimbatore unit? | the answer is in a **bullet list** |
| 28 | Who reports security incidents and to whom? | needs **two sentences** combined |
| 29 | Is the Coimbatore unit open during Diwali? | needs **reasoning** across two sentences in the scan |

A good result: most of A right, all of B refused, and honest refusals (not wrong answers) on C.
