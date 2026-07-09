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

### Account Lockout and Recovery

Accounts are automatically locked after 5 consecutive failed login
attempts within a 15-minute window and can only be unlocked by IT Help
Desk after identity verification via a callback to the employee's
on-file phone number. Self-service password reset via the identity
provider is available for accounts not currently locked.

### Shared and Service Accounts

Shared accounts are prohibited except for a documented list of legacy
systems approved by the security team, each of which must have a named
owner responsible for credential rotation. Service account credentials
(used by applications, not humans) must be stored in the company secrets
vault and rotated at least every 90 days, more frequently than the
180-day human password rotation cycle.

## Multi-Factor Authentication

MFA is mandatory for all accounts accessing email, VPN, source control,
and financial systems. Employees have a 7-day grace period after
onboarding to enroll a hardware key or authenticator app before access is
suspended. SMS-based MFA is disallowed for privileged accounts.

### Approved MFA Methods

Approved MFA factors are hardware security keys (YubiKey, company
standard), authenticator apps (Okta Verify, Google Authenticator), and
platform biometrics (Touch ID, Windows Hello) tied to a registered
device. Privileged accounts, defined as those with administrative access
to production systems, cloud infrastructure, or the identity provider,
must use a hardware security key as at least one enrolled factor.

### Lost MFA Device Procedure

Employees who lose an MFA device must report it to IT within 1 hour, the
same window required for lost devices generally, so that the factor can
be revoked and a temporary bypass code issued for use only after a
manager confirms the employee's identity via video call.

## Incident Reporting

Suspected security incidents (phishing, lost devices, unauthorized
access) must be reported to security@company.example within 1 hour of
discovery via the #security-incidents Slack channel or the 24/7 hotline.
The security team commits to an initial triage response within 30
minutes during business hours and 2 hours after hours.

### Incident Severity Levels

Incidents are triaged into three severities: SEV1 (active data breach or
production outage with security impact, requiring executive
notification within 1 hour of confirmation), SEV2 (contained incident
with limited scope, such as a single compromised account, requiring
notification to the security leadership within 4 hours), and SEV3 (low-
impact events such as a single blocked phishing attempt, tracked but not
requiring escalation). Severity is assigned by the on-call security
engineer during initial triage.

### Post-Incident Review

Every SEV1 and SEV2 incident requires a blameless post-incident review
within 5 business days of resolution, documenting root cause, timeline,
and remediation actions, distributed to affected teams and retained by
the security team for at least 2 years.

## Data Classification

Data is classified into three tiers: Public, Internal, and Restricted.
Restricted data (customer PII, financial records, source code) may only
be stored in approved systems and must be encrypted at rest using AES-256.
Transferring Restricted data via personal email or USB drives is a
terminable offense.

### Data in Transit

Restricted and Internal data must be encrypted in transit using TLS 1.2
or higher for any transmission across a public network. Internal-only
tools that transmit data solely within the corporate VPN or private cloud
network are exempt from this requirement but must still use encrypted
protocols where technically feasible.

### Approved Storage and SaaS Tools

Restricted data may only be stored in systems that have completed a
security review by the IT Security team, listed in the internal Approved
Tools catalog. Employees requesting a new SaaS tool that will handle
Restricted or Internal data must submit a Vendor Security Questionnaire
at least 10 business days before intended use; unapproved tools
("shadow IT") handling company data are subject to immediate
deprovisioning.

## Device Security

All laptops must have disk encryption, endpoint detection software, and
automatic screen lock after 5 minutes of inactivity. Lost or stolen
devices must be reported within 1 hour so IT can initiate remote wipe.

### Bring Your Own Device (BYOD)

Personal mobile devices used to access company email or Slack must be
enrolled in the company's mobile device management (MDM) system, which
enforces a device passcode and the ability to remotely wipe the
company data partition only (not personal data) upon loss or offboarding.
BYOD is not permitted for access to source control or production
infrastructure; those require a company-issued laptop per the Remote Work
Policy.

### Patch Management

Company-issued laptops must apply critical operating system security
patches within 7 days of release and all other patches within 30 days.
Devices that fall out of compliance are automatically flagged by the
endpoint management system and may have network access restricted until
remediated.

### Physical Security

Office badge access is role-based and logged; badges must not be shared
or lent to other employees or visitors. Visitors must be signed in at
reception, badged as "Visitor," and escorted in any area where Restricted
data is accessible on unlocked screens.

## Removable Media and Data Loss Prevention

USB drives, external hard drives, and other removable media are
disabled by default on company laptops via endpoint management; an
exception may be requested through the IT portal for a specific business
need and is time-limited to 90 days. Data Loss Prevention (DLP) tooling
monitors for Restricted data leaving approved systems (uploads to
unapproved cloud storage, mass email attachments containing patterns
matching Social Security or credit card numbers) and automatically
blocks the action while alerting the security team.

## Third-Party and Vendor Security Assessments

Vendors that will access company systems or Restricted data must
complete a security questionnaire and, for Tier 1 vendors as defined in
the Data Privacy Policy's vendor risk tiering, an annual penetration
test summary review before onboarding and annually thereafter. High-risk
findings identified in a vendor assessment must be remediated by the
vendor within 90 days or the engagement is escalated to the vendor
management committee for a go/no-go decision.

## Security Awareness Refresher Training

Beyond the initial Security Awareness Training required within an
employee's first 14 days (per the Onboarding Guide), all employees
complete an annual refresher covering current phishing tactics, safe
data handling, and incident reporting procedures. Employees who fail two
consecutive quarterly simulated phishing tests are required to complete
additional one-on-one training with the security team within 10 business
days.

## Cloud Infrastructure Access Controls

Production cloud infrastructure access follows the principle of least
privilege: engineers receive read-only access by default, with
write/deploy access granted only for their team's owned services and
reviewed quarterly by engineering leadership. All production access,
whether human or automated, is logged and retained for at least 1 year
to support incident investigations.
