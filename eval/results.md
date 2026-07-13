# Evaluation Results

- Groundedness: 92.0%
- Citation accuracy: 84.0%
- Latency p50: 2248.8 ms
- Latency p95: 2442.2 ms

| ID | Question | Grounded | Citation OK | Expected Doc |
|---|---|---|---|---|
| q01 | How many PTO days do full-time employees accrue per year, and at what monthly rate? | True | True | pto-policy |
| q02 | How many unused PTO days can be carried over to the next calendar year, and by when must they be used? | True | True | pto-policy |
| q03 | After how many months of continuous employment are employees eligible to request remote work status? | True | True | remote-work-policy |
| q04 | What is the home office setup stipend for a newly approved fully-remote employee, and what additional monthly stipend do they receive? | True | True | remote-work-policy |
| q05 | Within how many calendar days must an employee submit an expense report through Expensify, and what happens if it is submitted after 60 days? | True | True | expense-policy |
| q06 | What gift value threshold triggers a disclosure requirement to a manager and Compliance under the Code of Conduct? | False | True | code-of-conduct |
| q07 | What are the minimum password length and rotation requirements under the Information Security Policy? | True | True | security-policy |
| q08 | How quickly must a suspected security incident be reported, and what is the initial triage response time during business hours? | True | True | security-policy |
| q09 | Within how many days of joining must new hires complete Security Awareness Training and Anti-Harassment Training? | True | True | onboarding-guide |
| q10 | What milestones make up the new hire check-in schedule, and what benefit becomes available at the 90-day mark? | True | False | onboarding-guide |
| q11 | What percentage of the employee premium does the company cover on the Silver medical plan, and what is its individual deductible? | True | False | benefits-overview |
| q12 | How much does the company match on employee 401(k) contributions? | True | True | benefits-overview |
| q13 | How far in advance must domestic business travel be booked through the Concur travel portal? | True | True | travel-policy |
| q14 | What is the nightly hotel spend cap for domestic travel, and how much higher is it in designated high-cost cities? | True | True | travel-policy |
| q15 | How long is customer support ticket data containing personal data retained after case closure? | True | True | data-privacy-policy |
| q16 | Within how many hours must a manager escalate a harassment complaint to HR after receiving or witnessing it? | True | True | anti-harassment-policy |
| q17 | How many paid holidays does the company observe per year, and how many additional floating holidays do full-time employees receive? | True | True | holiday-schedule |
| q18 | How often do formal performance reviews occur, and what rating triggers a Performance Improvement Plan (PIP)? | True | True | performance-review-policy |
| q19 | How long does a Performance Improvement Plan (PIP) run, and how often are check-ins held during it? | True | True | performance-review-policy |
| q20 | How quickly must a departing employee return company equipment after their last day, and what happens if it is not returned within 30 days? | True | False | equipment-offboarding-policy |
| q21 | How many weeks of paid parental leave do birthing parents receive, and at what percentage of base salary? | False | False | parental-leave-policy |
| q22 | What is the annual professional development budget for a full-time employee, and does unused budget carry over to the next year? | True | True | learning-development-policy |
| q23 | What must an employee do before using a personal mobile device to access company email, and what does that system enforce? | True | True | acceptable-use-policy |
| q24 | Within how many days of hire must employees complete inclusion and anti-bias training, and how often afterward? | True | True | dei-policy |
| q25 | Within how many hours must a workplace injury or near-miss be reported? | True | True | workplace-safety-policy |