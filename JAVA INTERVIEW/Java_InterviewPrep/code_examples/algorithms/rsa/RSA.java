package rsa;
import java.math.BigInteger;
import java.util.Scanner;
public class RSA {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter a prime number p: ");
        BigInteger p = sc.nextBigInteger();
        System.out.print("Enter a prime number q: ");
        BigInteger q = sc.nextBigInteger();

        BigInteger n = p.multiply(q);
        BigInteger phi = p.subtract(BigInteger.ONE).multiply(q.subtract(BigInteger.ONE));

        System.out.print("Enter a message to encrypt: ");
        BigInteger msg =sc.nextBigInteger();

        System.out.print("Enter a public key: ");
        BigInteger e =sc.nextBigInteger();

        BigInteger cipher = msg.modPow(e, n);
        System.out.println("Encrypted text: "+cipher);
        BigInteger d = e.modInverse(phi);
        BigInteger decrypted = cipher.modPow(d, n);
        System.out.println("Drcrypted message: "+decrypted);
    
    }
}
