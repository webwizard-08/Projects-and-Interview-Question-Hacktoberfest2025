import java.security.SecureRandom;
import java.util.Scanner;

public class PasswordGenerator {
    private static final String UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    private static final String LOWER = "abcdefghijklmnopqrstuvwxyz";
    private static final String DIGITS = "0123456789";
    private static final String SYMBOLS = "!@#$%^&*()-_+=<>?";

    public static String generate(int length) {
        String all = UPPER + LOWER + DIGITS + SYMBOLS;
        SecureRandom rand = new SecureRandom();
        StringBuilder sb = new StringBuilder(length);

        // ensure at least one char from each group for stronger passwords
        sb.append(UPPER.charAt(rand.nextInt(UPPER.length())));
        sb.append(LOWER.charAt(rand.nextInt(LOWER.length())));
        sb.append(DIGITS.charAt(rand.nextInt(DIGITS.length())));
        sb.append(SYMBOLS.charAt(rand.nextInt(SYMBOLS.length())));

        for (int i = 4; i < length; i++) {
            sb.append(all.charAt(rand.nextInt(all.length())));
        }

        // simple shuffle
        char[] pwd = sb.toString().toCharArray();
        for (int i = pwd.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            char tmp = pwd[i];
            pwd[i] = pwd[j];
            pwd[j] = tmp;
        }
        return new String(pwd);
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter desired password length (min 6): ");
        int len = sc.nextInt();
        if (len < 6) {
            System.out.println("Length too short. Using 6.");
            len = 6;
        }
        String password = generate(len);
        System.out.println("Generated password: " + password);
        sc.close();
    }
}
