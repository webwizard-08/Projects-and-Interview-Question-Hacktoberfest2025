public class IntegerSqrt {
    // returns floor(sqrt(x)) for x >= 0
    public static int isqrt(int x) {
        if (x < 0) throw new IllegalArgumentException("x must be non-negative");
        int l = 0, r = x, ans = 0;
        while (l <= r) {
            int m = l + (r - l) / 2;
            long sq = 1L * m * m;
            if (sq == x) return m;
            if (sq < x) { ans = m; l = m + 1; }
            else r = m - 1;
        }
        return ans;
    }

    public static void main(String[] args) {
        System.out.println("isqrt(0) = " + isqrt(0));
        System.out.println("isqrt(1) = " + isqrt(1));
        System.out.println("isqrt(15) = " + isqrt(15)); // 3
        System.out.println("isqrt(16) = " + isqrt(16)); // 4
        System.out.println("isqrt(27) = " + isqrt(27)); // 5
    }
}
