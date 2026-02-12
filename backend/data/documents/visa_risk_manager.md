# Visa Risk Manager (VRM)

## Introduction
Visa Risk Manager (VRM) is a comprehensive fraud management platform that empowers issuers to create and manage rules to decline high-risk transactions in real-time. It acts as the "brain" deciding whether to authorize or block a payment based on risk signals.

## Core Components

### 1. Real-Time Decisioning (RTD)
The engine that evaluates transactions in milliseconds during the authorization flow.
- **Rules-Based**: "If transaction > $500 AND country != US, then DECLINE."
- **Score-Based**: "If Visa Advanced Authorization (VAA) Score < 10 (High Risk), then DECLINE."

### 2. Visa Advanced Authorization (VAA)
A machine learning model that scores every transaction from 1 (High Risk) to 99 (Low Risk).
- Analyzes 500+ data elements (location, device ID, spending pattern, time of day).
- VRM uses this score as a key input for its rules.

### 3. Case Management
When a rule triggers a "Review" decision instead of an outright "Decline."
- Analysts review suspicious transactions in a web portal.
- Can contact cardholders to verify activity via SMS/Email (Visa Consumer Authentication Service).

## Creating a Rule (Example)
An issuer wants to block all gambling transactions on corporate cards.
1. **Criteria**: `Merchant Category Code (MCC) = 7995` (Gambling).
2. **Bin Range**: `400000 - 499999` (Corporate BINs).
3. **Action**: `Decline`.
4. **Response Code**: `59` (Suspected Fraud).

## Integration
VRM is typically accessed via:
- **Visa Online (VOL)**: Web-based GUI for non-technical risk managers.
- **VRM APIs**: Programmatic access to manage rules and lists.
  - `POST /vrm/rules`: Create a new rule.
  - `GET /vrm/lists/blocklist`: Retrieve currently blocked card numbers.

## Strategy Optimization
VRM provides back-testing capabilities.
- **"What If" Analysis**: Run a proposed rule against historical transaction data to see how many legitimate customers would have been declined (False Positives).
