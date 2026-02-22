#!/usr/bin/env python3
"""
Decision Tree with leaf retrieval functionality.
"""


class Decision_Tree:
    """
    Decision Tree class.
    """

    def __init__(self, root=None):
        """
        Initialize the Decision Tree.

        Args:
            root (Node): root node of the tree
        """
        self.root = root

    def get_leaves(self):
        """
        Return all leaves of the tree.
        """
        return self.root.get_leaves_below()


class Leaf:
    """
    Leaf node of a Decision Tree.
    """

    def __init__(self, value, depth=None):
        """
        Initialize a Leaf.

        Args:
            value (int): value stored in leaf
            depth (int): depth of leaf
        """
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def get_leaves_below(self):
        """
        Return list containing this leaf.
        """
        return [self]


class Node:
    """
    Internal node of a Decision Tree.
    """

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None,
                 depth=None, is_root=False):
        """
        Initialize a Node.

        Args:
            feature (int): feature index
            threshold (float): split threshold
            left_child (Node/Leaf): left subtree
            right_child (Node/Leaf): right subtree
            depth (int): depth of node
            is_root (bool): whether node is root
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False

    def get_leaves_below(self):
        """
        Return all leaves below this node.
        """
        leaves = []

        if self.left_child is not None:
            leaves.extend(self.left_child.get_leaves_below())

        if self.right_child is not None:
            leaves.extend(self.right_child.get_leaves_below())

        return leaves
