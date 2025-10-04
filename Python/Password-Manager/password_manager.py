"""
Secure Password Manager
A command-line password manager with encryption using the cryptography library.
Features:
- Store passwords securely with encryption
- Generate strong random passwords
- Search and retrieve passwords
- Update and delete entries
- Master password protection
"""

import os
import json
import getpass
import secrets
import string
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import base64


class PasswordManager:
    def __init__(self):
        self.data_file = "passwords.enc"
        self.key_file = "key.key"
        self.passwords = {}
        self.cipher = None
        
    def generate_key(self, master_password):
        """Generate encryption key from master password"""
        salt = b'hacktoberfest2025'  # In production, use random salt and store it
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_password.encode()))
        return key
    
    def initialize_cipher(self, master_password):
        """Initialize the encryption cipher"""
        key = self.generate_key(master_password)
        self.cipher = Fernet(key)
    
    def load_passwords(self):
        """Load and decrypt passwords from file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    encrypted_data = f.read()
                decrypted_data = self.cipher.decrypt(encrypted_data)
                self.passwords = json.loads(decrypted_data.decode())
                return True
            except Exception as e:
                print(f"❌ Error loading passwords: {str(e)}")
                print("This might be due to an incorrect master password.")
                return False
        return True
    
    def save_passwords(self):
        """Encrypt and save passwords to file"""
        try:
            json_data = json.dumps(self.passwords, indent=2)
            encrypted_data = self.cipher.encrypt(json_data.encode())
            with open(self.data_file, 'wb') as f:
                f.write(encrypted_data)
            return True
        except Exception as e:
            print(f"❌ Error saving passwords: {str(e)}")
            return False
    
    def generate_password(self, length=16, use_symbols=True, use_numbers=True, use_uppercase=True):
        """Generate a strong random password"""
        characters = string.ascii_lowercase
        if use_uppercase:
            characters += string.ascii_uppercase
        if use_numbers:
            characters += string.digits
        if use_symbols:
            characters += string.punctuation
        
        password = ''.join(secrets.choice(characters) for _ in range(length))
        return password
    
    def add_password(self, service, username, password=None):
        """Add a new password entry"""
        if password is None:
            print("\n🔐 Generate a strong password? (y/n): ", end='')
            choice = input().strip().lower()
            if choice == 'y':
                length = int(input("Password length (default 16): ") or 16)
                password = self.generate_password(length)
                print(f"✅ Generated password: {password}")
            else:
                password = getpass.getpass("Enter password: ")
        
        self.passwords[service] = {
            'username': username,
            'password': password
        }
        
        if self.save_passwords():
            print(f"✅ Password for '{service}' saved successfully!")
        return True
    
    def get_password(self, service):
        """Retrieve a password entry"""
        if service in self.passwords:
            entry = self.passwords[service]
            print(f"\n📋 Service: {service}")
            print(f"👤 Username: {entry['username']}")
            print(f"🔑 Password: {entry['password']}")
            return entry
        else:
            print(f"❌ No password found for '{service}'")
            return None
    
    def list_services(self):
        """List all stored services"""
        if not self.passwords:
            print("📭 No passwords stored yet.")
            return
        
        print("\n📚 Stored Services:")
        print("-" * 40)
        for i, service in enumerate(self.passwords.keys(), 1):
            print(f"{i}. {service}")
        print("-" * 40)
    
    def update_password(self, service):
        """Update an existing password entry"""
        if service not in self.passwords:
            print(f"❌ No password found for '{service}'")
            return False
        
        print(f"\nUpdating password for '{service}'")
        print("1. Generate new password")
        print("2. Enter new password manually")
        choice = input("Choose option (1/2): ").strip()
        
        if choice == '1':
            length = int(input("Password length (default 16): ") or 16)
            new_password = self.generate_password(length)
            print(f"✅ Generated password: {new_password}")
        else:
            new_password = getpass.getpass("Enter new password: ")
        
        self.passwords[service]['password'] = new_password
        
        if self.save_passwords():
            print(f"✅ Password for '{service}' updated successfully!")
        return True
    
    def delete_password(self, service):
        """Delete a password entry"""
        if service not in self.passwords:
            print(f"❌ No password found for '{service}'")
            return False
        
        confirm = input(f"⚠️  Are you sure you want to delete '{service}'? (yes/no): ").strip().lower()
        if confirm == 'yes':
            del self.passwords[service]
            if self.save_passwords():
                print(f"✅ Password for '{service}' deleted successfully!")
            return True
        else:
            print("❌ Deletion cancelled.")
            return False
    
    def search_passwords(self, query):
        """Search for passwords by service name"""
        results = [service for service in self.passwords.keys() if query.lower() in service.lower()]
        
        if results:
            print(f"\n🔍 Found {len(results)} result(s):")
            print("-" * 40)
            for service in results:
                entry = self.passwords[service]
                print(f"📋 Service: {service}")
                print(f"👤 Username: {entry['username']}")
                print("-" * 40)
        else:
            print(f"❌ No results found for '{query}'")


def print_banner():
    """Print welcome banner"""
    banner = """
    ╔═══════════════════════════════════════╗
    ║     🔐 SECURE PASSWORD MANAGER 🔐     ║
    ║     Hacktoberfest 2025 Project        ║
    ╚═══════════════════════════════════════╝
    """
    print(banner)


def print_menu():
    """Print main menu"""
    menu = """
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
    """
    print(menu)


def main():
    """Main function to run the password manager"""
    print_banner()
    
    pm = PasswordManager()
    
    # Master password authentication
    if os.path.exists(pm.data_file):
        print("🔐 Enter your master password to unlock the vault:")
        master_password = getpass.getpass("Master Password: ")
    else:
        print("🎉 Welcome! Let's set up your password manager.")
        print("📌 Please create a strong master password.")
        master_password = getpass.getpass("Create Master Password: ")
        confirm_password = getpass.getpass("Confirm Master Password: ")
        
        if master_password != confirm_password:
            print("❌ Passwords don't match. Exiting.")
            return
        
        print("✅ Master password created successfully!")
    
    pm.initialize_cipher(master_password)
    
    if not pm.load_passwords():
        print("❌ Failed to unlock vault. Exiting.")
        return
    
    print("✅ Vault unlocked successfully!")
    
    # Main loop
    while True:
        print_menu()
        choice = input("Enter your choice (1-8): ").strip()
        
        if choice == '1':
            service = input("\n📝 Service name (e.g., Gmail, Facebook): ").strip()
            username = input("👤 Username/Email: ").strip()
            pm.add_password(service, username)
            
        elif choice == '2':
            service = input("\n🔍 Service name: ").strip()
            pm.get_password(service)
            
        elif choice == '3':
            pm.list_services()
            
        elif choice == '4':
            query = input("\n🔎 Search query: ").strip()
            pm.search_passwords(query)
            
        elif choice == '5':
            service = input("\n✏️  Service name to update: ").strip()
            pm.update_password(service)
            
        elif choice == '6':
            service = input("\n🗑️  Service name to delete: ").strip()
            pm.delete_password(service)
            
        elif choice == '7':
            print("\n🔑 Password Generator")
            length = int(input("Length (default 16): ") or 16)
            password = pm.generate_password(length)
            print(f"✅ Generated password: {password}")
            
        elif choice == '8':
            print("\n👋 Thank you for using Secure Password Manager!")
            print("🔒 Your passwords are safely encrypted.")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")
        print("\n" * 2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
