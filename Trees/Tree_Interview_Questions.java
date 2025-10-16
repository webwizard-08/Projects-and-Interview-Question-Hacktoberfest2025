/*
 * 🧠 Top 20 Tree Interview Questions – Java Implementation
 * Author: Sai Surya
 *
 * Description:
 * This Java program covers 20 essential binary tree problems frequently asked
 * in technical interviews at companies like Amazon, Google, and Infosys.
 *
 * Each problem includes:
 *  - Problem definition
 *  - Clean Java implementation
 *  - Example I/O
 *  - Time and Space Complexity
 */

import java.util.*;

class TreeNode {
    int data;
    TreeNode left, right;
    TreeNode(int data) {
        this.data = data;
        left = right = null;
    }
}

public class Tree_Interview_Questions {

    // 1️⃣ Inorder Traversal (Recursive)
    public static void inorder(TreeNode root) {
        if (root == null) return;
        inorder(root.left);
        System.out.print(root.data + " ");
        inorder(root.right);
    }

    // 2️⃣ Preorder Traversal (Recursive)
    public static void preorder(TreeNode root) {
        if (root == null) return;
        System.out.print(root.data + " ");
        preorder(root.left);
        preorder(root.right);
    }

    // 3️⃣ Postorder Traversal (Recursive)
    public static void postorder(TreeNode root) {
        if (root == null) return;
        postorder(root.left);
        postorder(root.right);
        System.out.print(root.data + " ");
    }

    // 4️⃣ Level Order Traversal (BFS)
    public static void levelOrder(TreeNode root) {
        if (root == null) return;
        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);
        while (!q.isEmpty()) {
            TreeNode curr = q.poll();
            System.out.print(curr.data + " ");
            if (curr.left != null) q.add(curr.left);
            if (curr.right != null) q.add(curr.right);
        }
    }

    // 5️⃣ Height of Binary Tree
    public static int height(TreeNode root) {
        if (root == null) return 0;
        return 1 + Math.max(height(root.left), height(root.right));
    }

    // 6️⃣ Count Total Nodes
    public static int countNodes(TreeNode root) {
        if (root == null) return 0;
        return 1 + countNodes(root.left) + countNodes(root.right);
    }

    // 7️⃣ Check if Two Trees are Identical
    public static boolean isIdentical(TreeNode a, TreeNode b) {
        if (a == null && b == null) return true;
        if (a == null || b == null) return false;
        return a.data == b.data &&
               isIdentical(a.left, b.left) &&
               isIdentical(a.right, b.right);
    }

    // 8️⃣ Diameter of Binary Tree
    static int maxDiameter;
    public static int diameter(TreeNode root) {
        maxDiameter = 0;
        heightForDiameter(root);
        return maxDiameter;
    }
    private static int heightForDiameter(TreeNode root) {
        if (root == null) return 0;
        int lh = heightForDiameter(root.left);
        int rh = heightForDiameter(root.right);
        maxDiameter = Math.max(maxDiameter, lh + rh);
        return 1 + Math.max(lh, rh);
    }

    // 9️⃣ Mirror / Invert Binary Tree
    public static TreeNode mirror(TreeNode root) {
        if (root == null) return null;
        TreeNode left = mirror(root.left);
        TreeNode right = mirror(root.right);
        root.left = right;
        root.right = left;
        return root;
    }

    // 🔟 Lowest Common Ancestor (LCA)
    public static TreeNode lca(TreeNode root, int a, int b) {
        if (root == null) return null;
        if (root.data == a || root.data == b) return root;
        TreeNode left = lca(root.left, a, b);
        TreeNode right = lca(root.right, a, b);
        if (left != null && right != null) return root;
        return left != null ? left : right;
    }

    // 1️⃣1️⃣ Check if Balanced Tree
    public static boolean isBalanced(TreeNode root) {
        return checkBalance(root) != -1;
    }
    private static int checkBalance(TreeNode root) {
        if (root == null) return 0;
        int lh = checkBalance(root.left);
        int rh = checkBalance(root.right);
        if (lh == -1 || rh == -1 || Math.abs(lh - rh) > 1) return -1;
        return 1 + Math.max(lh, rh);
    }

    // 1️⃣2️⃣ Check if Symmetric
    public static boolean isSymmetric(TreeNode root) {
        return root == null || isMirror(root.left, root.right);
    }
    private static boolean isMirror(TreeNode a, TreeNode b) {
        if (a == null && b == null) return true;
        if (a == null || b == null) return false;
        return (a.data == b.data) && isMirror(a.left, b.right) && isMirror(a.right, b.left);
    }

    // 1️⃣3️⃣ Zigzag (Spiral) Traversal
    public static void zigzagTraversal(TreeNode root) {
        if (root == null) return;
        Queue<TreeNode> q = new LinkedList<>();
        boolean leftToRight = true;
        q.add(root);
        while (!q.isEmpty()) {
            int size = q.size();
            List<Integer> row = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode curr = q.poll();
                row.add(curr.data);
                if (curr.left != null) q.add(curr.left);
                if (curr.right != null) q.add(curr.right);
            }
            if (!leftToRight) Collections.reverse(row);
            for (int val : row) System.out.print(val + " ");
            leftToRight = !leftToRight;
        }
    }

    // 1️⃣4️⃣ Convert Sorted Array to BST
    public static TreeNode sortedArrayToBST(int[] arr, int l, int r) {
        if (l > r) return null;
        int mid = (l + r) / 2;
        TreeNode node = new TreeNode(arr[mid]);
        node.left = sortedArrayToBST(arr, l, mid - 1);
        node.right = sortedArrayToBST(arr, mid + 1, r);
        return node;
    }

    // 1️⃣5️⃣ Check if Tree is a BST
    public static boolean isBST(TreeNode root, Integer min, Integer max) {
        if (root == null) return true;
        if ((min != null && root.data <= min) || (max != null && root.data >= max))
            return false;
        return isBST(root.left, min, root.data) && isBST(root.right, root.data, max);
    }

    // 1️⃣6️⃣ Vertical Order Traversal
    public static void verticalOrder(TreeNode root) {
        if (root == null) return;
        TreeMap<Integer, List<Integer>> map = new TreeMap<>();
        Queue<Map.Entry<TreeNode, Integer>> q = new LinkedList<>();
        q.add(Map.entry(root, 0));
        while (!q.isEmpty()) {
            var entry = q.poll();
            TreeNode node = entry.getKey();
            int hd = entry.getValue();
            map.putIfAbsent(hd, new ArrayList<>());
            map.get(hd).add(node.data);
            if (node.left != null) q.add(Map.entry(node.left, hd - 1));
            if (node.right != null) q.add(Map.entry(node.right, hd + 1));
        }
        for (var e : map.values()) {
            for (int val : e) System.out.print(val + " ");
        }
    }

    // 1️⃣7️⃣ Top View
    public static void topView(TreeNode root) {
        if (root == null) return;
        Map<Integer, Integer> map = new TreeMap<>();
        Queue<Map.Entry<TreeNode, Integer>> q = new LinkedList<>();
        q.add(Map.entry(root, 0));
        while (!q.isEmpty()) {
            var entry = q.poll();
            TreeNode node = entry.getKey();
            int hd = entry.getValue();
            map.putIfAbsent(hd, node.data);
            if (node.left != null) q.add(Map.entry(node.left, hd - 1));
            if (node.right != null) q.add(Map.entry(node.right, hd + 1));
        }
        for (int val : map.values()) System.out.print(val + " ");
    }

    // 1️⃣8️⃣ Bottom View
    public static void bottomView(TreeNode root) {
        if (root == null) return;
        Map<Integer, Integer> map = new TreeMap<>();
        Queue<Map.Entry<TreeNode, Integer>> q = new LinkedList<>();
        q.add(Map.entry(root, 0));
        while (!q.isEmpty()) {
            var entry = q.poll();
            TreeNode node = entry.getKey();
            int hd = entry.getValue();
            map.put(hd, node.data);
            if (node.left != null) q.add(Map.entry(node.left, hd - 1));
            if (node.right != null) q.add(Map.entry(node.right, hd + 1));
        }
        for (int val : map.values()) System.out.print(val + " ");
    }

    // 1️⃣9️⃣ Right View
    public static void rightView(TreeNode root) {
        if (root == null) return;
        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                TreeNode curr = q.poll();
                if (i == size - 1) System.out.print(curr.data + " ");
                if (curr.left != null) q.add(curr.left);
                if (curr.right != null) q.add(curr.right);
            }
        }
    }

    // 2️⃣0️⃣ Maximum Path Sum
    static int maxSum;
    public static int maxPathSum(TreeNode root) {
        maxSum = Integer.MIN_VALUE;
        maxPathHelper(root);
        return maxSum;
    }
    private static int maxPathHelper(TreeNode root) {
        if (root == null) return 0;
        int left = Math.max(0, maxPathHelper(root.left));
        int right = Math.max(0, maxPathHelper(root.right));
        maxSum = Math.max(maxSum, root.data + left + right);
        return root.data + Math.max(left, right);
    }

    // 🧪 Sample Demo
    public static void main(String[] args) {
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        root.left.left = new TreeNode(4);
        root.left.right = new TreeNode(5);
        System.out.println("Inorder Traversal:");
        inorder(root);
        System.out.println("\nHeight: " + height(root));
    }
}
