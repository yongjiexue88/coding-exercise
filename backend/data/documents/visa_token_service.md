# Visa Token Service (VTS) Technical Guide

## Overview
The Visa Token Service (VTS) replaces sensitive cardholder information, such as the Primary Account Number (PAN), with a unique digital identifier called a **Token**. This enhances security for digital payments by removing the actual card number from the transaction flow.

## Key Concepts

### 1. Token
A 16-digit number that looks like a PAN but is worthless if stolen.
- **Domain Restricted**: A token is often bound to a specific device (e.g., iPhone), merchant (e.g., Netflix), or channel (e.g., eCommerce only).
- **Format Preserving**: Compatible with existing ISO 8583 payment messages.

### 2. Token Requestor (TR)
The entity requesting a token.
- **TR-TSM**: Trusted Service Manager (e.g., Apple Pay, Google Pay).
- **COF Merchant**: Card-on-File merchant (e.g., Amazon, Uber).
- **Issuer Wallet**: Bank's own mobile app.

### 3. Token Vault
A secure database managed by Visa that maps Tokens back to the underlying PANs.

## The Tokenization Lifecycle

### Step 1: Provisioning (Creation)
1. **Enrollment**: Consumer adds card to a wallet or merchant app.
2. **Token Request**: TR sends PAN + consumer data to Visa.
3. **ID&V**: Identification & Verification. The Issuer (Bank) validates the user (via OTP, App login, or call center).
4. **Issuance**: Visa generates a Token and stores the PAN-Token mapping in access Token Vault.
5. **Activation**: The Token is delivered to the device/merchant and is ready for use.

### Step 2: Transaction Processing
1. **Payment**: Consumer pays using the Token.
2. **Authorization Request**: Acquirer sends the Token to Visa.
3. **De-Tokenization**: Visa looks up the Token in the Vault, retrieves the real PAN, and forwards the request to the Issuer.
4. **Authorization Response**: Issuer approves/declines the real PAN.
5. **Re-Tokenization**: Visa replaces the PAN with the Token again before sending the response back to the Merchant/Acquirer.

## Benefits

- **Risk Reduction**: If a merchant database is hacked, thieves only get Tokens, not real card numbers.
- **Lifecycle Management**: If a physical card is lost/replaced, the Bank can update the backend mapping in the Vault. The user does *not* need to update their card on file (unlike with PANs).
- **Higher Approval Rates**: Issuers trust tokenized transactions more, leading to fewer false declines.
