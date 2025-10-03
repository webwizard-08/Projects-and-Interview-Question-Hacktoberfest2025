# 🔐 Secure Password Manager

A command-line password manager with strong encryption built with Python. This project demonstrates secure password storage, encryption, and password generation.

## ✨ Features

- 🔒 **Encrypted Storage**: All passwords are encrypted using Fernet symmetric encryption
- 🔑 **Master Password Protection**: Single master password to access all stored passwords
- 🎲 **Password Generator**: Generate strong, random passwords with customizable length
- 📝 **CRUD Operations**: Create, Read, Update, and Delete password entries
- 🔍 **Search Functionality**: Quickly find passwords by service name
- 👤 **Username Storage**: Store both usernames and passwords for each service
- 💾 **Persistent Storage**: Passwords are saved securely to an encrypted file

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Install required dependencies:**

```bash
pip install cryptography
```

2. **Run the password manager:**

```bash
python password_manager.py
```

### First Time Setup

1. When you run the program for the first time, you'll be asked to create a **master password**
2. This master password will be used to encrypt and decrypt all your stored passwords
3. **⚠️ Important**: Don't forget your master password! There's no way to recover it.

## 📖 How to Use

### Main Menu Options

1. **📝 Add new password** - Store a new password for a service
   - Enter service name (e.g., Gmail, Facebook, Twitter)
   - Enter username/email
   - Choose to generate a strong password or enter your own

2. **🔍 Get password** - Retrieve a stored password
   - Enter the service name
   - View username and password

3. **📚 List all services** - Display all stored service names

4. **🔎 Search passwords** - Search for services by name
   - Enter a search query
   - View matching services with usernames

5. **✏️ Update password** - Change an existing password
   - Enter service name
   - Choose to generate new password or enter manually

6. **🗑️ Delete password** - Remove a password entry
   - Enter service name
   - Confirm deletion

7. **🔑 Generate password** - Create a strong random password
   - Specify desired length
   - Password is generated but not stored

8. **🚪 Exit** - Close the application securely

## 🔐 Security Features

### Encryption
- Uses **Fernet** (symmetric encryption) from the cryptography library
- Passwords are encrypted before saving to disk
- Master password is never stored - it's used to derive the encryption key

### Key Derivation
- Uses **PBKDF2** (Password-Based Key Derivation Function 2)
- 100,000 iterations for strong key derivation
- SHA-256 hashing algorithm

### Password Generation
- Uses Python's `secrets` module for cryptographically strong random passwords
- Includes uppercase, lowercase, numbers, and symbols
- Customizable length (default: 16 characters)

## 📁 File Structure

```
Password-Manager/
├── password_manager.py    # Main application file
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── passwords.enc         # Encrypted passwords (created on first use)
└── .gitignore           # Git ignore file
```

## 🛡️ Best Practices

1. **Choose a strong master password**
   - Use at least 12 characters
   - Mix uppercase, lowercase, numbers, and symbols
   - Don't use common words or patterns

2. **Keep your master password safe**
   - Write it down in a secure location
   - Don't share it with anyone
   - Don't use the same password elsewhere

3. **Regular backups**
   - Backup the `passwords.enc` file regularly
   - Store backups in a secure location

4. **Use generated passwords**
   - Let the tool generate strong passwords for you
   - Use unique passwords for each service

## ⚠️ Important Notes

- This is a **local** password manager - data is stored on your computer
- The encrypted file (`passwords.enc`) contains all your passwords
- If you lose your master password, there's **no way to recover** your data
- For production use, consider adding additional features like:
  - Random salt generation and storage
  - Password strength meter
  - Clipboard integration
  - Backup/export functionality
  - Two-factor authentication

## 🤝 Contributing

This project is part of **Hacktoberfest 2025**! Contributions are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contributions

- Add password strength indicator
- Implement clipboard integration
- Add export/import functionality
- Create a GUI version
- Add password history tracking
- Implement automatic backups
- Add two-factor authentication

## 📝 Example Usage

```bash
$ python password_manager.py

    ╔═══════════════════════════════════════╗
    ║     🔐 SECURE PASSWORD MANAGER 🔐     ║
    ║     Hacktoberfest 2025 Project        ║
    ╚═══════════════════════════════════════╝

🎉 Welcome! Let's set up your password manager.
📌 Please create a strong master password.
Create Master Password: ********
Confirm Master Password: ********
✅ Master password created successfully!
✅ Vault unlocked successfully!

    ╔═══════════════════════════════════════╗
    ║              MAIN MENU                ║
    ╠═══════════════════════════════════════╣
    ║  1. 📝 Add new password               ║
    ║  2. 🔍 Get password                   ║
    ║  3. 📚 List all services              ║
    ║  4. 🔎 Search passwords               ║
    ║  5. ✏️  Update password                ║
    ║  6. 🗑️  Delete password                ║
    ║  7. 🔑 Generate password              ║
    ║  8. 🚪 Exit                           ║
    ╚═══════════════════════════════════════╝

Enter your choice (1-8): 1

📝 Service name (e.g., Gmail, Facebook): GitHub
👤 Username/Email: myemail@example.com

🔐 Generate a strong password? (y/n): y
Password length (default 16): 20
✅ Generated password: K#9mP$xL2@vN4qR8tY&z
✅ Password for 'GitHub' saved successfully!
```

## 📄 License

This project is open source and available for educational purposes.

## 🎃 Hacktoberfest 2025

This project is created as part of Hacktoberfest 2025 to encourage open-source contributions and help developers learn about:
- Encryption and security
- Password management
- Python programming
- Command-line applications

Happy coding! 🚀
