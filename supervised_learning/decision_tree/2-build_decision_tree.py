#!/usr/bin/env python3
"""
Decision Tree representation with printable structure.
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

    def __str__(self):
        """
        Return string representation of the tree.
        """
        return self.root.__str__()


class Leaf:
    """
    Leaf node of a Decision Tree.
    """

    def __init__(self, value, depth=None):
        """
        Initialize a Leaf.

        Args:
            value (int): value stored in leaf
            depth (int): depth of leaf in tree
        """
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def __str__(self):
        """
        Return string representation of leaf.
        """
        return f"-> leaf [value={self.value}]"


class Node:
    """
    Internal node of a Decision Tree.
    """

    def __init__(self, feature, threshold, left_child=None,
                 right_child=None, depth=None, is_root=False):
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

    def left_child_add_prefix(self, text):
        """
        Add prefix formatting to left child string.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "    |  " + line + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Add prefix formatting to right child string.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "       " + line + "\n"
        return new_text

    def __str__(self):
        """
        Return string representation of node and its children.
        """
        if self.is_root:
            text = (
                f"root [feature={self.feature}, "
                f"threshold={self.threshold}]"
            )
        else:
            text = (
                f"-> node [feature={self.feature}, "
                f"threshold={self.threshold}]"
            )

        left_text = self.left_child_add_prefix(
            str(self.left_child)
        )
        right_text = self.right_child_add_prefix(
            str(self.right_child)
        )

        return text + "\n" + left_text + right_text.rstrip("\n")
