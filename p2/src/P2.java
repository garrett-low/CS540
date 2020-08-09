import java.io.*;
import java.text.DecimalFormat;
import java.util.*;

public class P2 {

    // This code uses all 9 features to train the decision tree
    public static int numAttr = 9;
    private static final String outputPath = "./output.txt";
    private static final DecimalFormat fourPlaces = new DecimalFormat("0.0000");

    public static void main(String[] args) {
        createFile(outputPath);

        try (FileWriter writer = new FileWriter(outputPath)) {
            // Import training set
            List<ArrayList<Integer>> trainData = cleanAndTransformData("./breast-cancer-wisconsin.data");

            // Q1 - count imported data
            q1ImportDataCheck(trainData, writer);

            // Q2 - entropy at root
            q2EntropyAtRoot(trainData, writer);

            // Q3 - decision stump
            q3Stump(trainData, 8, writer);

            // Build training tree
            DecTreeNode root = buildTree(trainData);
            printTree(root);
            printSideways(root, "");

            // TODO: need to implement method for choosing specific features out of all features of train dataset
            // i.e.features that are generated using your student id
            // TODO: need to implement method for classifying test data given the root of the decision tree
            // TODO: need to implement method for finding maximum depth of the tree
            // TODO: need to implement method for pruning the tree to have a fixed maximum depth
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void q1ImportDataCheck(List<ArrayList<Integer>> trainData, FileWriter writer) throws IOException {
        int benignCount = 0;
        int malignantCount = 0;
        for (ArrayList<Integer> patientData : trainData) {
            Integer classLabel = patientData.get(9);
            if (classLabel == 2) {
                benignCount++;
            } else {
                malignantCount++;
            }
        }
        writer.write("Q1 - benign, malignant: " + benignCount + "," + malignantCount + "\n");
        System.out.println("Q1 - benign, malignant: " + benignCount + "," + malignantCount);
    }

    /*
     * This method cleans the train dataset, removes rows with missing values,
     * and ignores first id column, since that's not used during training.
     */
    public static List<ArrayList<Integer>> cleanAndTransformData(String file) {

        List<ArrayList<Integer>> cleanData = new ArrayList<ArrayList<Integer>>();

        try {
            Scanner scan = new Scanner(new File(file));

            while (scan.hasNextLine()) {
                String line = scan.nextLine();
                // if the line doesn't contain '?', we should include it into our clean dataset
                if (line.indexOf('?') == -1) {
                    String[] split = line.split(",");

                    ArrayList<Integer> instance = new ArrayList<Integer>();
                    // ignore id column, it's not used for training
                    // convert string values to integer values
                    for (int i = 1; i < split.length; i++) {
                        instance.add(Integer.parseInt(split[i]));
                    }
                    cleanData.add(instance);
                }
            }
            scan.close();
        } catch (FileNotFoundException e) {
            System.out.println("File with the name " + file + " cannot be read!");
        }
        return cleanData;
    }

    /*
     * Method for computing entropy.
     */
    public static double entropy(double p0) {
        if (p0 == 0 || p0 == 1) return 0;

        double p1 = 1 - p0;
        return -(p0 * Math.log(p0) / Math.log(2) + p1 * Math.log(p1) / Math.log(2));
    }

    /*
     * Method for computing information gain.
     */
    public static double informationGain(List<ArrayList<Integer>> dataSet, int feature, int threshold) {

        int dataSize = dataSet.size();
        // first computing H(Y)
        int count = 0;
        for (List<Integer> data : dataSet) {
            if (data.get(data.size() - 1) == 2) { // if label equals 2, last column is label
                count++;
            }
        }
        double Hy = entropy((double) count / dataSize);

        // computing H(Y|X), conditional entropy
        double Hyx = 0;
        int countLess = 0;
        int countGreater = 0;
        int countLessAndPositive = 0;
        int countGreaterAndPositive = 0;
        for (List<Integer> data : dataSet) {
            if (data.get(feature) <= threshold) {
                countLess++;
                if (data.get(data.size() - 1) == 2) countLessAndPositive++; // last column is label
            } else {
                countGreater++;
                if (data.get(data.size() - 1) == 2) countGreaterAndPositive++;
            }
        }
        double prob1 = (double) countLess / dataSize;
        double prob2 = (double) countGreater / dataSize;
        if (prob1 > 0) {
            Hyx = Hyx + prob1 * entropy(((double) countLessAndPositive) / countLess);
        }
        if (prob2 > 0) {
            Hyx = Hyx + prob2 * entropy(((double) countGreaterAndPositive) / countGreater);
        }
        // return difference between entropy and conditional entropy
        return Hy - Hyx;
    }

    /*
     * Method for building a decision tree using the given dataset.
     * It returns a pointer to the root node.
     */
    private static DecTreeNode buildTree(List<ArrayList<Integer>> dataSet) {
        int numData = dataSet.size();
        int bestAttr = -1;
        int bestThreshold = Integer.MIN_VALUE;
        double bestScore = Double.NEGATIVE_INFINITY;
        boolean leaf = false;
        DecTreeNode node = null;

        // if node isn't leaf node, compute the best split for it
        if (!leaf) {
            for (int j = 0; j < numAttr; j++) {
                for (int i = 1; i < 11; i++) {
                    double score = informationGain(dataSet, j, i);
                    if (score > bestScore) {
                        bestScore = score;
                        bestAttr = j;
                        bestThreshold = i;
                    }
                }
            }
            if (bestScore == 0) {
                leaf = true;
            }

            // split the entire # of instances into two groups based on the threshold value (<= and >)
            List<ArrayList<Integer>> leftList = new ArrayList<ArrayList<Integer>>();
            List<ArrayList<Integer>> rightList = new ArrayList<ArrayList<Integer>>();
            for (ArrayList<Integer> data : dataSet) {
                if (data.get(bestAttr) <= bestThreshold) {
                    leftList.add(data);
                } else {
                    rightList.add(data);
                }
            }

            if (leftList.size() == 0 || rightList.size() == 0) {
                leaf = true;
            }
            // if node is not leaf, create left and right children
            if (!leaf) {
                node = new DecTreeNode(-1, bestAttr, bestThreshold);
                node.left = buildTree(leftList);
                node.right = buildTree(rightList);
            }
        }
        // if node is leaf, need to count # of instances with labels 2 and 4
        if (leaf) {
            int count = 0;
            for (List<Integer> data : dataSet) {
                if (data.get(data.size() - 1) == 2)
                    count += 1;
            }

            if (count >= numData - count) {
                node = new DecTreeNode(2, -1, -1); // assign label 2 to the leaf node
            } else {
                node = new DecTreeNode(4, -1, -1); // assign label 4 to the leaf node
            }
        }
        return node;
    }

    private static void q2EntropyAtRoot(List<ArrayList<Integer>> trainData, FileWriter writer) throws IOException {
        int dataSize = trainData.size();
        // first computing H(Y)
        int count = 0;
        for (List<Integer> data : trainData) {
            if (data.get(data.size() - 1) == 2) { // if label equals 2, last column is label
                count++;
            }
        }
        double Hy = entropy((double) count / dataSize);
        writer.write("Q2 - entropy at root before split: " + fourPlaces.format(Hy) + "\n");
        System.out.println("Q2 - entropy at root before split: " + fourPlaces.format(Hy));
    }

    /*
     * Method for building a decision tree using the given dataset.
     * It returns a pointer to the root node.
     */
    private static void q3Stump(List<ArrayList<Integer>> dataSet, int featureStump, FileWriter writer) throws IOException {
        int numData = dataSet.size();
        int bestThreshold = Integer.MIN_VALUE;
        double bestScore = Double.NEGATIVE_INFINITY;
        DecTreeNode node = null;

        for (int i = 1; i < 11; i++) {
            double score = informationGain(dataSet, featureStump, i);
            if (score > bestScore) {
                bestScore = score;
                bestThreshold = i;
            }
        }

        // split the entire # of instances into two groups based on the threshold value (<= and >)
        List<ArrayList<Integer>> leftList = new ArrayList<ArrayList<Integer>>();
        List<ArrayList<Integer>> rightList = new ArrayList<ArrayList<Integer>>();
        for (ArrayList<Integer> data : dataSet) {
            if (data.get(featureStump) <= bestThreshold) {
                leftList.add(data);
            } else {
                rightList.add(data);
            }
        }

        int leftBenignCount = 0, leftMalignantCount = 0;
        for (ArrayList<Integer> patient : leftList) {
            if (patient.get(patient.size() - 1) == 2) {
                leftBenignCount++;
            } else {
                leftMalignantCount++;
            }
        }

        int rightBenignCount = 0, rightMalignantCount = 0;
        for (ArrayList<Integer> patient : rightList) {
            if (patient.get(patient.size() - 1) == 2) {
                rightBenignCount++;
            } else {
                rightMalignantCount++;
            }
        }

        int[] q3array = {leftBenignCount, rightBenignCount, leftMalignantCount, rightMalignantCount};

        writer.write("Q3 - stump positive and negative counts (above-benign, below-benign, above-malignant, below-malignant): " + Arrays.toString(q3array) + "\n");
        System.out.println("Q3 - stump positive and negative counts (above-benign, below-benign, above-malignant, below-malignant): " + Arrays.toString(q3array));

        writer.write("Q4 - information gain: " + fourPlaces.format(bestScore) + "\n");
        System.out.println("Q4 - information gain: " + fourPlaces.format(bestScore));
    }

    // TODO: need to implement method for printing the tree in the correct format
    /*
     * Method which given the root of the Decision Tree, prints it
     * in the format specified at the webpage for P2 assignment.
     */
    private static void printTree(DecTreeNode node) {
        System.out.println(preOrderTraversal(node));
    }

    private static String preOrderTraversal(DecTreeNode node) {
        String preOrderString = "";

        if (node != null) {
            preOrderString = "if (x" + String.valueOf(node.feature) + String.valueOf(node.threshold) + String.valueOf(node.classLabel) + "," + preOrderTraversal(node.left) + preOrderTraversal(node.right);
        }

        return preOrderString;
    }

    private static void printSideways(DecTreeNode current, String indent) {
        if (current != null) {
            printSideways(current.right, indent + "    ");
            System.out.println(indent + "[feat:" + current.feature + ", thresh:" + current.threshold + ", label:" + current.classLabel +"]");
            printSideways(current.left, indent + "    ");
        }
    }

    private static void createFile(String outputPath) {
        try {
            File myObj = new File(outputPath);
            if (myObj.createNewFile()) {
                System.out.println("File created: " + myObj.getName());
            } else {
                System.out.println("File already exists.");
            }
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }
}