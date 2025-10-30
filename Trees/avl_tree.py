"""
AVL Tree Implementation - Self-Balancing Binary Search Tree

AVL Tree Properties:
1. It is a BST (Binary Search Tree)
2. The difference between heights of left and right subtrees for any node (balance factor)
   cannot be more than 1
3. Balance factor = Height of left subtree - Height of right subtree

Time Complexities:
- Insertion: O(log n)
- Deletion: O(log n)
- Search: O(log n)
- Space Complexity: O(n)

Applications:
1. Database indexing
2. Self-balancing in-memory data structures
3. Memory allocation
"""

class AVLNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1  # New node is initially added at leaf

class AVLTree:
    def __init__(self):
        self.root = None
    
    def height(self, node):
        """Get height of the node"""
        if not node:
            return 0
        return node.height
    
    def balance_factor(self, node):
        """Get balance factor of the node"""
        if not node:
            return 0
        return self.height(node.left) - self.height(node.right)
    
    def update_height(self, node):
        """Update height of the node"""
        if not node:
            return
        node.height = max(self.height(node.left),
                         self.height(node.right)) + 1
    
    def right_rotate(self, y):
        """
        Right rotation
             y                               x
            / \     Right Rotation          /  \
           x   T3   - - - - - - - >      T1    y
          / \                                  / \
         T1  T2                              T2  T3
        """
        x = y.left
        T2 = x.right
        
        # Perform rotation
        x.right = y
        y.left = T2
        
        # Update heights
        self.update_height(y)
        self.update_height(x)
        
        return x
    
    def left_rotate(self, x):
        """
        Left rotation
            x                               y
           /  \                            /  \
          T1   y   Left Rotation        x    T3
              / \  - - - - - - - >    / \
            T2  T3                   T1  T2
        """
        y = x.right
        T2 = y.left
        
        # Perform rotation
        y.left = x
        x.right = T2
        
        # Update heights
        self.update_height(x)
        self.update_height(y)
        
        return y
    
    def insert(self, root, key):
        """Insert a key into AVL tree"""
        # Perform normal BST insertion
        if not root:
            return AVLNode(key)
        
        if key < root.key:
            root.left = self.insert(root.left, key)
        elif key > root.key:
            root.right = self.insert(root.right, key)
        else:  # Equal keys not allowed
            return root
        
        # Update height of ancestor node
        self.update_height(root)
        
        # Get balance factor
        balance = self.balance_factor(root)
        
        # Left Left Case
        if balance > 1 and key < root.left.key:
            return self.right_rotate(root)
        
        # Right Right Case
        if balance < -1 and key > root.right.key:
            return self.left_rotate(root)
        
        # Left Right Case
        if balance > 1 and key > root.left.key:
            root.left = self.left_rotate(root.left)
            return self.right_rotate(root)
        
        # Right Left Case
        if balance < -1 and key < root.right.key:
            root.right = self.right_rotate(root.right)
            return self.left_rotate(root)
        
        return root
    
    def delete(self, root, key):
        """Delete a key from AVL tree"""
        if not root:
            return root
        
        # Perform standard BST delete
        if key < root.key:
            root.left = self.delete(root.left, key)
        elif key > root.key:
            root.right = self.delete(root.right, key)
        else:
            # Node with only one child or no child
            if not root.left:
                temp = root.right
                root = None
                return temp
            elif not root.right:
                temp = root.left
                root = None
                return temp
            
            # Node with two children
            temp = self._get_min_value_node(root.right)
            root.key = temp.key
            root.right = self.delete(root.right, temp.key)
        
        # If tree had only one node
        if not root:
            return root
        
        # Update height
        self.update_height(root)
        
        # Get balance factor
        balance = self.balance_factor(root)
        
        # Left Left Case
        if balance > 1 and self.balance_factor(root.left) >= 0:
            return self.right_rotate(root)
        
        # Left Right Case
        if balance > 1 and self.balance_factor(root.left) < 0:
            root.left = self.left_rotate(root.left)
            return self.right_rotate(root)
        
        # Right Right Case
        if balance < -1 and self.balance_factor(root.right) <= 0:
            return self.left_rotate(root)
        
        # Right Left Case
        if balance < -1 and self.balance_factor(root.right) > 0:
            root.right = self.right_rotate(root.right)
            return self.left_rotate(root)
        
        return root
    
    def _get_min_value_node(self, node):
        """Get node with minimum key value in AVL tree"""
        current = node
        while current.left:
            current = current.left
        return current
    
    def inorder_traversal(self, root):
        """Inorder traversal of AVL tree"""
        if not root:
            return []
        return (self.inorder_traversal(root.left) +
                [root.key] +
                self.inorder_traversal(root.right))

def main():
    """Example usage with test cases"""
    avl_tree = AVLTree()
    root = None
    
    # Test case 1: Insert elements
    keys = [10, 20, 30, 40, 50, 25]
    print("Inserting keys:", keys)
    for key in keys:
        root = avl_tree.insert(root, key)
        print(f"Inserted {key}, Inorder traversal:", avl_tree.inorder_traversal(root))
    
    # Test case 2: Delete elements
    delete_keys = [30, 20]
    print("\nDeleting keys:", delete_keys)
    for key in delete_keys:
        root = avl_tree.delete(root, key)
        print(f"Deleted {key}, Inorder traversal:", avl_tree.inorder_traversal(root))

if __name__ == "__main__":
    main()
