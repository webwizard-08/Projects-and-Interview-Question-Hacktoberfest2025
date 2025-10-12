/**
 * 🌳 Top 20 Tree Interview Questions – Java Implementation
 * Author: Sai Surya
 * 
 * Description:
 * This Java program contains 20 essential tree-based coding problems
 * frequently asked in interviews at companies like FAANG, TCS, Infosys, and Amazon.
 * It covers binary tree and BST concepts, traversals, recursion, and advanced tree logic.
 */

import java.util.*;

public class Tree_Interview_Questions {

    // Basic structure for a Binary Tree node
    static class Node {
        int data;
        Node left, right;

        Node(int value) {
            data = value;
            left = right = null;
        }
    }

    // 1️⃣ Inorder Traversal (Left -> Root -> Right)
    public static void inorder(Node root) {
        if (root == null) return;
        inorder(root.left);
        System.out.print(root.data + " ");
        inorder(root.right);
    }

    // 2️⃣ Preorder Traversal (Root -> Left -> Right)
    public static void preorder(Node root) {
        if (root == null) return;
        System.out.print(root.data + " ");
        preorder(root.left);
        preorder(root.right);
    }

    // 3️⃣ Postorder Traversal (Left -> Right -> Root)
    public static void postorder(Node root) {
        if (root == null) return;
        postorder(root.left);
        postorder(root.right);
        System.out.print(root.data + " ");
    }

    // 4️⃣ Level Order Traversal (BFS)
    public static void levelOrder(Node root) {
        if (root == null) return;
        Queue<Node> q = new LinkedList<>();
        q.add(root);
        while (!q.isEmpty()) {
            Node temp = q.poll();
            System.out.print(temp.data + " ");
            if (temp.left != null) q.add(temp.left);
            if (temp.right != null) q.add(temp.right);
        }
    }

    // 5️⃣ Find Height of a Binary Tree
    public static int height(Node root) {
        if (root == null) return 0;
        return 1 + Math.max(height(root.left), height(root.right));
    }

    // 6️⃣ Count Total Nodes
    public static int countNodes(Node root) {
        if (root == null) return 0;
        return 1 + countNodes(root.left) + countNodes(root.right);
    }

    // 7️⃣ Find Maximum Value in Tree
    public static int findMax(Node root) {
        if (root == null) return Integer.MIN_VALUE;
        return Math.max(root.data, Math.max(findMax(root.left), findMax(root.right)));
    }

    // 8️⃣ Check if Two Trees are Identical
    public static boolean isIdentical(Node a, Node b) {
        if (a == null && b == null) return true;
        if (a == null || b == null) return false;
        return (a.data == b.data) && isIdentical(a.left, b.left) && isIdentical(a.right, b.right);
    }

    // 9️⃣ Check if Tree is Symmetric
    public static boolean isSymmetric(Node root) {
        return isMirror(root, root);
    }

    private static boolean isMirror(Node t1, Node t2) {
        if (t1 == null && t2 == null) return true;
        if (t1 == null || t2 == null) return false;
        return (t1.data == t2.data)
                && isMirror(t1.left, t2.right)
                && isMirror(t1.right, t2.left);
    }

    // 🔟 Find Diameter of Binary Tree
    static int maxDiameter = 0;
    public static int diameter(Node root) {
        maxDiameter = 0;
        heightForDiameter(root);
        return maxDiameter;
    }

    private static int heightForDiameter(Node root) {
        if (root == null) return 0;
        int lh = heightForDiameter(root.left);
        int rh = heightForDiameter(root.right);
        maxDiameter = Math.max(maxDiameter, lh + rh);
        return 1 + Math.max(lh, rh);
    }

    // 1️⃣1️⃣ Lowest Common Ancestor (LCA)
    public static Node LCA(Node root, int n1, int n2) {
        if (root == null) return null;
        if (root.data == n1 || root.data == n2) return root;

        Node left = LCA(root.left, n1, n2);
        Node right = LCA(root.right, n1, n2);

        if (left != null && right != null) return root;
        return (left != null) ? left : right;
    }

    // 1️⃣2️⃣ Check if a Tree is a Valid BST
    public static boolean isValidBST(Node root, long min, long max) {
        if (root == null) return true;
        if (root.data <= min || root.data >= max) return false;
        return isValidBST(root.left, min, root.data) && isValidBST(root.right, root.data, max);
    }

    // 1️⃣3️⃣ Sum of All Nodes
    public static int sumOfNodes(Node root) {
        if (root == null) return 0;
        return root.data + sumOfNodes(root.left) + sumOfNodes(root.right);
    }

    // 1️⃣4️⃣ Print All Root-to-Leaf Paths
    public static void printPaths(Node root, List<Integer> path) {
        if (root == null) return;
        path.add(root.data);
        if (root.left == null && root.right == null)
            System.out.println(path);
        else {
            printPaths(root.left, path);
            printPaths(root.right, path);
        }
        path.remove(path.size() - 1);
    }

    // 1️⃣5️⃣ Maximum Path Sum in a Binary Tree
    static int maxSum = Integer.MIN_VALUE;
    public static int maxPathSum(Node root) {
        maxSum = Integer.MIN_VALUE;
        pathSum(root);
        return maxSum;
    }

    private static int pathSum(Node root) {
        if (root == null) return 0;
        int left = Math.max(0, pathSum(root.left));
        int right = Math.max(0, pathSum(root.right));
        maxSum = Math.max(maxSum, left + right + root.data);
        return root.data + Math.max(left, right);
    }

    // 1️⃣6️⃣ Construct Tree from Inorder and Preorder
    static int preIndex = 0;
    public static Node buildTree(int[] inorder, int[] preorder, int inStart, int inEnd) {
        if (inStart > inEnd) return null;
        Node root = new Node(preorder[preIndex++]);
        int inIndex = search(inorder, inStart, inEnd, root.data);
        root.left = buildTree(inorder, preorder, inStart, inIndex - 1);
        root.right = buildTree(inorder, preorder, inIndex + 1, inEnd);
        return root;
    }

    private static int search(int[] arr, int start, int end, int value) {
        for (int i = start; i <= end; i++)
            if (arr[i] == value) return i;
        return -1;
    }

    // 1️⃣7️⃣ Serialize and Deserialize a Binary Tree
    public static String serialize(Node root) {
        if (root == null) return "#,";
        return root.data + "," + serialize(root.left) + serialize(root.right);
    }

    public static Node deserialize(Queue<String> q) {
        if (q.isEmpty()) return null;
        String val = q.poll();
        if (val.equals("#")) return null;
        Node root = new Node(Integer.parseInt(val));
        root.left = deserialize(q);
        root.right = deserialize(q);
        return root;
    }

    // 1️⃣8️⃣ Convert Sorted Array to Balanced BST
    public static Node sortedArrayToBST(int[] arr, int start, int end) {
        if (start > end) return null;
        int mid = (start + end) / 2;
        Node root = new Node(arr[mid]);
        root.left = sortedArrayToBST(arr, start, mid - 1);
        root.right = sortedArrayToBST(arr, mid + 1, end);
        return root;
    }

    // 1️⃣9️⃣ Vertical Order Traversal
    public static void verticalOrder(Node root) {
        if (root == null) return;
        TreeMap<Integer, List<Integer>> map = new TreeMap<>();
        Queue<Pair> q = new LinkedList<>();
        q.add(new Pair(root, 0));
        while (!q.isEmpty()) {
            Pair temp = q.poll();
            map.computeIfAbsent(temp.hd, k -> new ArrayList<>()).add(temp.node.data);
            if (temp.node.left != null) q.add(new Pair(temp.node.left, temp.hd - 1));
            if (temp.node.right != null) q.add(new Pair(temp.node.right, temp.hd + 1));
        }
        for (List<Integer> list : map.values()) {
            System.out.println(list);
        }
    }

    static class Pair {
        Node node;
        int hd;
        Pair(Node n, int h) {
            node = n;
            hd = h;
        }
    }

    // 2️⃣0️⃣ Check if a Binary Tree is Height Balanced
    public static boolean isBalanced(Node root) {
        return checkHeight(root) != -1;
    }

    private static int checkHeight(Node root) {
        if (root == null) return 0;
        int left = checkHeight(root.left);
        int right = checkHeight(root.right);
        if (left == -1 || right == -1 || Math.abs(left - right) > 1) return -1;
        return Math.max(left, right) + 1;
    }

    // 🧪 Driver code for demonstration
    public static void main(String[] args) {
        Node root = new Node(1);
        root.left = new Node(2);
        root.right = new Node(3);
        root.left.left = new Node(4);
        root.left.right = new Node(5);

        System.out.println("Inorder Traversal:");
        inorder(root);

        System.out.println("\nHeight of Tree: " + height(root));
        System.out.println("Is Balanced: " + isBalanced(root));
    }
}
