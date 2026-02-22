#!/usr/bin/env python3
"""Decision Tree depth computation module."""

import numpy as np


class Node:
    """Represents an internal node of a decision tree."""

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None,
                 is_root=False, depth=0):
        """
        Initialize a Node.

        Parameters:
        feature: feature index used for splitting
        threshold: threshold value for splitting
        left_child: left subtree
        right_child: right subtree
        is_root: boolean indicating if node is root
        depth: depth of the node in the tree
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the maximum depth among all nodes
        below this node (including leaves).
        """
        if self.is_leaf:
            return self.depth

        left_depth = (
            self.left_child.max_depth_below()
            if self.left_child else self.depth
        )

        right_depth = (
            self.right_child.max_depth_below()
            if self.right_child else self.depth
        )

        return max(left_depth, right_depth)


class Leaf(Node):
    """Represents a leaf node of a decision tree."""

    def __init__(self, value, depth=None):
        """
        Initialize a Leaf.

        Parameters:
        value: predicted value of the leaf
        depth: depth of the leaf in the tree
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns the depth of the leaf."""
        return self.depth


class Decision_Tree():
    """Decision Tree class."""

    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        """
        Initialize a Decision Tree.

        Parameters:
        max_depth: maximum allowed depth
        min_pop: minimum population for splitting
        seed: random seed
        split_criterion: splitting method
        root: root node
        """
        self.rng = np.random.default_rng(seed)

        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)

        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Returns the maximum depth of the tree."""
        return self.root.max_depth_below()
