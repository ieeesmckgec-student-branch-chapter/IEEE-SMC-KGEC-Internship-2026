# ConsenSys - Medical Insurance Claiming DApp

**Project Title:** ConsenSys (Medical Insurance Claiming DApp)
**Team Member:** Jayesh J. Pandey

## Project Description
This decentralized application (DApp) is built for managing and verifying medical insurance claims on the Ethereum blockchain. The workflow involves multiple parties ensuring a transparent and verifiable claiming process:
1. **Patient** logs in, uploads medical/lab test bills, and submits them for insurance. Notifications are sent to the hospital and lab admin.
2. **Hospital admin** logs in, verifies, and approves the bills. This approval is stored securely on the smart contract.
3. **Lab admin** approves the lab test bills, which is also stored on the smart contract.
4. Once both admins approve, notifications are sent to the **Insurance admin**.
5. The **Insurance admin** can check for approvals from the hospital and lab, calculate the claim amount, and process the claim.

The `HealthCare.sol` contract handles the core business logic, while the React front-end in the `Web-client` folder communicates with the deployed smart contract and allows specific users to log in and interact with the system.

## Technologies Used
* **Smart Contracts:** Solidity
* **Blockchain Network:** Ethereum (Ganache for local testing)
* **Frontend:** React.js, Web3.js
* **Development Framework:** Truffle, Remix IDE
* **Wallet integration:** MetaMask

## Setup & Installation Instructions

### Prerequisites
- Node.js installed
- MetaMask extension installed in your browser
- Ganache CLI (or Ganache GUI)

### Smart Contract Deployment
1. Copy and paste the contract code (`contracts/HealthCare.sol`) on https://remix.ethereum.org/
2. Run an instance of Ganache on your local machine and connect your MetaMask wallet to it. 
3. Add the first 3 accounts from Ganache to your MetaMask by importing their private keys and assign the following names:
   * **Account 1:** Hospital admin
   * **Account 2:** Lab admin
   * **Account 3:** Patient
4. In Remix, pass the Lab Admin's address (Account 2) as an argument in the constructor while deploying the contract.
5. Select `Injected Provider - MetaMask` (formerly Injected Web3) in the `Environment` field on Remix and ensure your MetaMask wallet is unlocked. This connects Remix to the Hospital admin account.
6. Deploy the contract.

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd Web-client
   ```
2. Install the required dependencies:
   ```bash
   npm install
   ```
3. Start the React development server:
   ```bash
   npm start
   ```

## Usage Details

1. With the frontend running and contract deployed, select **Account 3 (Patient)** in MetaMask.
2. Create a new medical record by filling out the form on the DApp (which calls the `newRecord` function).
3. Verify the record creation by querying the `_records` mapping with index 1 (via Remix or the DApp UI if implemented).
4. To sign the record, switch to **Account 1 (Hospital admin)** in MetaMask, enter the record's `_ID` in the `signRecord` function, and click on transact.
5. Repeat the signing step using **Account 2 (Lab Admin)** from MetaMask.
6. The record is now fully approved! You can verify this by checking the `_records` mapping again, where the `signatureCount` will have incremented.
7. *Note:* A patient cannot sign their own record, and an admin cannot sign the same record twice.

---

### Known issues:  
* The table on the React front end doesn't display the records created by the user (Issue #1).  
* Note: The primary focus of this project was the smart contract logic; the front-end serves as a basic UI for interaction.

**Original GitHub Repository:** [https://github.com/jayeshpandey01/ConsenSys](https://github.com/jayeshpandey01/ConsenSys)
