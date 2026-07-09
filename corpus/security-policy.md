---
doc_id: security-policy
title: Information Security Policy
---

# Information Security Policy

## Password Requirements

All company accounts require passwords of at least 14 characters,
including one uppercase letter, one number, and one symbol. Passwords
must be rotated every 180 days and may not repeat any of the last 10
passwords used. Password managers (1Password, company-provisioned) are
required for all engineering and finance staff.

## Multi-Factor Authentication

MFA is mandatory for all accounts accessing email, VPN, source control,
and financial systems. Employees have a 7-day grace period after
onboarding to enroll a hardware key or authenticator app before access is
suspended. SMS-based MFA is disallowed for privileged accounts.

## Incident Reporting

Suspected security incidents (phishing, lost devices, unauthorized
access) must be reported to security@company.example within 1 hour of
discovery via the #security-incidents Slack channel or the 24/7 hotline.
The security team commits to an initial triage response within 30
minutes during business hours and 2 hours after hours.

## Data Classification

Data is classified into three tiers: Public, Internal, and Restricted.
Restricted data (customer PII, financial records, source code) may only
be stored in approved systems and must be encrypted at rest using AES-256.
Transferring Restricted data via personal email or USB drives is a
terminable offense.

## Device Security

All laptops must have disk encryption, endpoint detection software, and
automatic screen lock after 5 minutes of inactivity. Lost or stolen
devices must be reported within 1 hour so IT can initiate remote wipe.
