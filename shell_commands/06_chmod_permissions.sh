#!/bin/bash

# Command: chmod (Change Mode)
# Description: Change file permissions and access rights for files and directories
#
# Permission Types:
#   r (read)    = 4
#   w (write)   = 2
#   x (execute) = 1
#
# User Categories:
#   u = user (owner)
#   g = group
#   o = others
#   a = all
#
# Common Permission Numbers:
#   777 = rwxrwxrwx (all permissions for everyone - NOT RECOMMENDED)
#   755 = rwxr-xr-x (owner: all, group/others: read & execute)
#   644 = rw-r--r-- (owner: read/write, group/others: read only)
#   600 = rw------- (owner: read/write, no access for others)
#   700 = rwx------ (owner: all permissions, no access for others)
#
# Usage Examples:
#   chmod 755 script.sh              # Set specific permissions using numbers
#   chmod +x script.sh               # Add execute permission for all
#   chmod u+x script.sh              # Add execute permission for owner only
#   chmod u+rw,g+r,o-r file.txt      # Multiple changes at once
#   chmod -R 755 directory/          # Recursively change directory permissions
#   chmod a-x file.txt               # Remove execute permission from all users
#
# Notes:
#   - Requires appropriate permissions to change file modes
#   - Be extremely careful with recursive (-R) flag, especially on system directories
#   - 777 permissions are a security risk - avoid unless absolutely necessary
#   - Execute permission on directories allows entering/accessing that directory
#   - Use 'ls -l' to view current permissions
#
# Security Best Practices:
#   - Scripts: 755 or 700
#   - Private files: 600
#   - Public files: 644
#   - Directories: 755 or 700

# Example 1: Make a script executable
echo "Example 1: Making a script executable"
# chmod +x myscript.sh
echo "Command: chmod +x myscript.sh"
echo "Result: Adds execute permission for user, group, and others"
echo ""

# Example 2: Set specific permissions using numbers
echo "Example 2: Setting permissions to 644 (rw-r--r--)"
# chmod 644 document.txt
echo "Command: chmod 644 document.txt"
echo "Result: Owner can read/write, others can only read"
echo ""

# Example 3: Remove write permission from group and others
echo "Example 3: Protecting a file"
# chmod go-w important.txt
echo "Command: chmod go-w important.txt"
echo "Result: Removes write permission from group and others"
echo ""

# Example 4: Make a directory and its contents accessible
echo "Example 4: Setting directory permissions"
# chmod -R 755 my_project/
echo "Command: chmod -R 755 my_project/"
echo "Result: Owner has full access, others can read and execute"
echo "Warning: -R applies changes recursively to all files/subdirectories"
echo ""

# Example 5: Secure a private key file
echo "Example 5: Securing SSH private key"
# chmod 600 ~/.ssh/id_rsa
echo "Command: chmod 600 ~/.ssh/id_rsa"
echo "Result: Only owner can read/write, required for SSH keys"
echo ""

# Example 6: View current permissions
echo "Example 6: Checking permissions"
# ls -l file.txt
echo "Command: ls -l file.txt"
echo "Result: Shows permissions like -rw-r--r-- (644)"
echo ""

# Quick Reference Table
echo "=== Quick Reference ==="
echo "Number | Permission | Symbolic"
echo "-------|------------|----------"
echo "  0    |    ---     | No permissions"
echo "  1    |    --x     | Execute only"
echo "  2    |    -w-     | Write only"
echo "  3    |    -wx     | Write and execute"
echo "  4    |    r--     | Read only"
echo "  5    |    r-x     | Read and execute"
echo "  6    |    rw-     | Read and write"
echo "  7    |    rwx     | All permissions"
echo ""

# Common Interview Questions:
echo "=== Interview Tips ==="
echo "Q: What does chmod 755 mean?"
echo "A: Owner (7=rwx), Group (5=r-x), Others (5=r-x)"
echo ""
echo "Q: How to make a script executable?"
echo "A: chmod +x script.sh"
echo ""
echo "Q: What's the difference between chmod 644 and 755?"
echo "A: 644: not executable, 755: executable"
echo ""

# Reference: https://man7.org/linux/man-pages/man1/chmod.1.html