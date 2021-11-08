package com.jml.clasifiers;

import java.util.ArrayList;
import java.util.List;

/**
 * A k-dimensional tree or K-d tree is a binary tree which partitions a space by organizing points in a K-dimensional
 * space. K-d trees can be used to solve generalized N-point problems e.g. K-nearest neighbors.
 */
public class KDTree {

    /**
     * Dimension of each point in the tree.
     */
    private int k;
    private PointNode root;

    /**
     *
     * @param k Dimension of each point in the k dimensional tree.
     */
    public KDTree(int k) {
        this.k = k;
    }


    /**
     * Inserts a point into this K-d tree.
     * @param point Point to insert.
     */
    public void insert(double[] point) {

        if(point.length!=k) {
            throw new IllegalArgumentException("Point does not have the same number of dimensions as the other points" +
                    "in the tree. Expecting point with dimension " + k + " but got " + point.length);
        }

        if(root==null) { // Then create a root.
            this.root = new PointNode(point);
        } else {
            PointNode current = root;
            int depth = 0;
            int axis;

            while(current!=null) {
                axis = depth % k;

                if(point[axis] < current.getValue(axis) ) {
                    if(current.left == null) {
                        current.left = new PointNode(point);
                        break;
                    } else {
                        current = current.left;
                    }
                } else {
                    if(current.right == null) {
                        current.right = new PointNode(point);
                        break;
                    } else {
                        current = current.right;
                    }
                }

                depth++;
            }
        }
    }


    /**
     * Gets the specified point from the tree if it exists.
     *
     * @param point Point to get from tree.
     * @return Returns null if no point in the tree matches the specified point. Returns the point
     */
    public double[] get(double[] point) {
        // TODO: Implementation.
        return null;
    }


    /**
     * Traverses this K-d tree in order.
     * @return Returns an arraylist containing the points from the in-order traversal of the tree.
     */
    public List<double[]> inOrder() {
        List<double[]> pointList = new ArrayList<>();
        return inOrder(root, pointList);
    }


    /**
     * Traverses this K-d tree in order.
     *
     * @param current Current Node in the traversal.
     * @param pointList Current list of points visited in the traversal.
     * @return Returns the points in order as a List.
     */
    private List<double[]> inOrder(PointNode current, List<double[]> pointList) {
        if(current!=null) {
            inOrder(current.left, pointList);
            pointList.add(current.get());
            inOrder(current.right, pointList);
        }

        return pointList;
    }


    /**
     * Node which contains a k-dimensional point.
     */
    class PointNode {
        double[] point;
        PointNode left, right, parent;


        /**
         * Creates a PointNode with a specified point.
         * @param point Point to insert into this PointNode.
         */
        public PointNode(double[] point) {
            if(point.length!=k) {
                throw new IllegalArgumentException("Point does not have the same number of dimensions as the other points" +
                        "in the tree. Expecting point with dimension " + k + " but got " + point.length);
            }

            this.point = point;
        }


        /**
         * Gets the point from the PointNode
         * @return Returns the point from the PointNode.
         */
        public double[] get() {
           return this.point;
        }


        /**
         * Gets the value of a point at the specified dimension.
         * @param dimension Dimension of point to get value of.
         *
         * @return
         */
        double getValue(int dimension) {
            return point[dimension];
        }
    }


    public static void main(String[] args) {
        double[][] values = {   {7, 2},
                                {5, 4},
                                {9, 6},
                                {4, 7},
                                {8, 1},
                                {2, 3}};

        KDTree tree = new KDTree(2);

        for(double[] point : values) {
            tree.insert(point);
        }

        List<double[]> vals = tree.inOrder();

        for(double[] v : vals) {
            for(double d : v) {
                System.out.print(d + ", ");
            }

            System.out.println();
        }
    }
}
