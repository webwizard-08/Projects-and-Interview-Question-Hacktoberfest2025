def encrypt_decrypt_char(plaintext_char):
    if plaintext_char.isalpha():
        first_alpha_letter = "a"

        if plaintext_char.isupper():
            first_alpha_letter = "A"

        old_char_position = ord(plaintext_char) - ord(first_alpha_letter)
        new_char_position = -(old_char_position + 1) % 26

        return chr(new_char_position + ord(first_alpha_letter))

    return plaintext_char

def encrypt_decrypt(text):
    new_text = ""

    for char in text:
        new_text += encrypt_decrypt_char(char)

    return new_text

plaintext = input("> ")

ciphertext = encrypt_decrypt(plaintext)
decrypted_plaintext = encrypt_decrypt(ciphertext)

print(f"Translated: {decrypted_plaintext}")
print(f"To: {ciphertext}")
