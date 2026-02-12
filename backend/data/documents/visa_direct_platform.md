# Visa Direct Platform Overview

## What is Visa Direct?
Visa Direct is a real-time push payments platform that allows businesses to send money directly to billions of financial accounts around the world. Unlike traditional card payments which "pull" funds from an account, Visa Direct instructions "push" funds to the recipient.

### Key Capabilities
- **Global Reach**: Connects to over 3 billion cards and bank accounts across 190+ countries.
- **Speed**:  Transactions are processed in real-time, often settling within 30 minutes.
- **Security**: Leverages Visa's global network security standards.

## Core Use Cases

### 1. Person-to-Person (P2P)
Enables individuals to send money to friends and family.
- **Example**: Splitting a dinner bill, sending rent money.
- **Flow**: Sender initiates transfer -> Originator (Bank/App) -> Visa Network -> Recipient Bank -> Recipient Account.

### 2. Business-to-Consumer (B2C) Disbursements
Businesses sending funds to customers.
- **Example**: Insurance claim payouts, gig economy worker earnings (Uber/DoorDash), gaming winnings, tax refunds.
- **Benefit**: Replaces slow checks and ACH transfers with instant availability.

### 3. Business-to-Business (B2B)
Small business payments and settlements.
- **Example**: Paying vendors, settling marketplace invoices.

### 4. Cross-Border Remittances
International money transfers between individuals.
- **Benefit**: Significantly faster and often cheaper than traditional SWIFT transfers.

## Technical Architecture

### The "Push" Payment Model (OCT)
Visa Direct utilizes the **Original Credit Transaction (OCT)** message. 
- **AFT (Account Funding Transaction)**: First leg. Pulls funds from the sender's source (e.g., their debit card or wallet balance).
- **OCT (Original Credit Transaction)**: Second leg. Pushes the funds to the recipient's Visa credential (PAN).

### API Overview
Integration is primarily done via REST APIs.

#### 1. Funds Transfer API
The core API for moving money.
- `POST /visadirect/fundstransfer/v1/pullfundstransactions`: Initiates AFT (funding).
- `POST /visadirect/fundstransfer/v1/pushfundstransactions`: Initiates OCT (disbursement).

**Sample Request Body (Push):**
```json
{
  "amount": "100.00",
  "currency": "USD",
  "recipientPrimaryAccountNumber": "4000123456789010",
  "senderName": "John Doe",
  "transactionIdentifier": "1234567890"
}
```

#### 2. Watch List Screening API
Required for cross-border and some domestic programs to ensure compliance with sanctions (OFAC, etc.).
- Checks sender and recipient against global watchlists before authorizing the transfer.

#### 3. Mobile Push Payment API
Optimized for mobile-to-mobile scenarios, often used in developing markets.

## Implementation Steps
1. **Register**: Sign up as an Originator or work with an Acquiring Bank.
2. **Sandbox**: Test APIs in the Visa Developer Center sandbox.
3. **Certification**: Pass Visa's security and operational certification.
4. **Go Live**: Production keys issued.

## Exceptions and Reversals
- Unlike pull payments, push payments are difficult to reverse.
- **Reversals**: Only allowed for technical errors (e.g., duplicate processing), not for fraud or disputes by the sender.
