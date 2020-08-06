import java.io.*;
import java.text.DecimalFormat;
import java.util.*;
import java.lang.Math;


// Todo: you need to change the activation function from relu (current) version to logistic, remember, not only the activation function, but the weight update part as well.

public class NN_student {

    // Todo: change hyper-parameters below, like MAX_EPOCHS, learning_rate, etc.

    private static final String COMMA_DELIMITER = ",";
    private static final String PATH_TO_TRAIN = "./src/mnist_train.csv";
    private static final String PATH_TO_TEST = "./src/mnist_test.csv";
    private static final String NEW_TEST = "./src/test_grlow.txt";
    private static final String outputPath = "./src/output.txt";
    private static final int MAX_EPOCHS = 10;
    static double learning_rate = 0.1;

    static double[][] wih = new double[392][785];
    static double[] who = new double[393];

    static String first_digit = "7";
    static String second_digit = "3";
    static Random rng = new Random();


    public static double[][] parseRecords(String file_path) throws IOException {
        double[][] records = new double[20000][786];
        try (BufferedReader br = new BufferedReader(new FileReader(file_path))) {
            String line;
            int k = 0;
            while ((line = br.readLine()) != null) {

                String[] string_values = line.split(COMMA_DELIMITER);
                if (!string_values[0].equals(first_digit) && !string_values[0].contentEquals(second_digit)) continue;
                if (first_digit.equals(string_values[0])) records[k][0] = 0.0; // label 0
                else records[k][0] = 1.0; // label 1
                for (int i = 1; i < string_values.length; i++) {
                    records[k][i] = Double.parseDouble(string_values[i]) / 255.0; // features
                }
                records[k][785] = 1.0;

                k += 1;
            }

            double[][] res = new double[k][786];
            for (int i = 0; i < k; i++) {
                System.arraycopy(records[i], 0, res[i], 0, 786);
            }
            return res;
        }

    }

    public static double[][] NewTest(String file_path) throws IOException {
        double[][] records = new double[20000][785];
        try (BufferedReader br = new BufferedReader(new FileReader(file_path))) {
            String line;
            int k = 0;
            while ((line = br.readLine()) != null) {

                String[] string_values = line.split(COMMA_DELIMITER);
                for (int i = 0; i < string_values.length; i++) {
                    records[k][i] = Double.parseDouble(string_values[i]) / 255.0; // features
                }
                records[k][784] = 1.0;

                k += 1;
            }

            double[][] res = new double[k][785];
            for (int i = 0; i < k; i++) {
                System.arraycopy(records[i], 0, res[i], 0, 785);
            }
            return res;
        }

    }

    public static double sigmoid(double weight, double bias) {
        return 1.0 / (1.0 + Math.exp(-1 * (weight + bias)));
    }

    public static double diff_sigmoid(double weight, double bias) {
        return sigmoid(weight, bias) * (1 - sigmoid(weight, bias));
    }

    public static double relu(double x) {
        if (x > 0) return x;
        else return 0.0;
    }

    public static double diff_relu(double x) {
        if (x > 0) return 1.0;
        else return 0.0;
    }


    public static void main(String[] args) {
        createFile(outputPath);

        try (FileWriter writer = new FileWriter(outputPath)) {
            writer.write("output file here\n");

            double[][] train = parseRecords(PATH_TO_TRAIN);
            double[][] test = parseRecords(PATH_TO_TEST);

            double[][] new_test = NewTest(NEW_TEST);

            int num_train = train.length;
            int num_test = new_test.length;

            for (int i = 0; i < wih.length; i++) {
                for (int j = 0; j < wih[0].length; j++) {
                    wih[i][j] = 2 * rng.nextDouble() - 1;
                }
            }
            for (int i = 0; i < who.length; i++) {
                who[i] = 2 * rng.nextDouble() - 1;
            }

            for (int i = 0; i < 392; i++) {
                wih[i][784] = rng.nextDouble();
            }

            who[392] = rng.nextDouble();

            for (int epoch = 1; epoch <= MAX_EPOCHS; epoch++) {
                double[] out_o = new double[num_train];
                double[][] out_h = new double[num_train][393];

                for (int ind = 0; ind < num_train; ind++) {
                    double[] row = train[ind];
                    double label = row[0];


                    //calc out_h[ind, :-1]
                    for (int i = 0; i < 392; i++) {
                        double s = 0.0;
                        for (int j = 0; j < 785; j++) {
                            s += wih[i][j] * row[j + 1];
                        }
//                    out_h[ind][i] = relu(s);
                        out_h[ind][i] = sigmoid(s, wih[i][784]);
                    }

                    //calc out_o[ind]
                    double s = 0.0;
                    out_o[392] = rng.nextDouble();
                    for (int i = 0; i < 392; i++) {
                        s += out_h[ind][i] * who[i];
                    }
//                out_o[ind] = 1.0 / (1.0 + Math.exp(-s));
                    out_o[ind] = sigmoid(s, who[392]);

                    //calc delta
                    double[] delta = new double[393];
                    for (int i = 0; i < 392; i++) {
                        delta[i] = diff_sigmoid(out_h[ind][i], wih[i][784]) * who[i] * (label - out_o[ind]);
                    }

                    //update wih
                    for (int i = 0; i < 392; i++) {
                        for (int j = 0; j < 785; j++) {
                            wih[i][j] += learning_rate * delta[i] * row[j + 1];
                        }
                    }

                    // update bias_hidden layer
                    for (int i = 0; i < 392; i++) {
                        wih[i][784] = learning_rate * delta[i];
                    }

                    //update who
                    for (int i = 0; i < 393; ++i) {
                        who[i] += learning_rate * (label - out_o[ind]) * out_h[ind][i];
                    }

                    // update bias output layer
                    for (int i = 0; i < 392; i++) {
                        who[392] =  learning_rate * (label - out_o[ind]);
                    }

                }

                //calc error
                double error = 0;
                for (int ind = 0; ind < num_train; ind++) {
                    double[] row = train[ind];
                    error += -row[0] * Math.log(out_o[ind]) - (1 - row[0]) * Math.log(1 - out_o[ind]);
                }

                //correct
                double correct = 0.0;
                for (int ind = 0; ind < num_train; ind++) {
                    if ((train[ind][0] == 1.0 && out_o[ind] >= 0.5) || (train[ind][0] == 0.0 && out_o[ind] < 0.5))
                        correct += 1.0;
                }
                double acc = correct / num_train;

                writer.write("Epoch: " + epoch + ", error: " + error + ", acc: " + acc + "\n");
                System.out.println("Epoch: " + epoch + ", error: " + error + ", acc: " + acc);
            }

            // Print weights, Q5 and Q6
            printWeights(writer, wih, who);

            // Todo: your new_test. Hint: use above 'new_test'
            double[][] test_act_ih = new double[new_test.length][392];
            double[] test_act_ho = new double[new_test.length];
            //calc test_act_ih[ind, :-1]
            for (int testIndex = 0; testIndex < num_test; testIndex++) {
                double[] row = new_test[testIndex];

                for (int i = 0; i < 392; ++i) {
                    double s = 0.0;
                    for (int j = 0; j < 200; ++j) {
                        s += row[i] * wih[i][j];
                    }

                    test_act_ih[testIndex][i] = sigmoid(s, wih[i][784]);
                }

                double s = 0.0;
                for (int i = 0; i < 392; i++) {
                    s += test_act_ih[testIndex][i] * who[i];
                }

                test_act_ho[testIndex] = sigmoid(s, who[392]);
            }

            // Q7 - print second layer activation on test set
            printTestSet(writer, test_act_ho, new_test);

        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }

    private static void printTestSet(FileWriter writer, double[] test_act_ho, double[][] test) throws IOException {
        DecimalFormat df = new DecimalFormat("0.0000");
        writer.write("Test set, second layer activation values\n");

        for (int i = 0; i < test_act_ho.length; i++) {
            writer.write(df.format(test_act_ho[i]));

            if (i < test_act_ho.length - 1) {
                writer.write(",");
            } else {
                writer.write("\n");
            }
        }

        DecimalFormat rounded = new DecimalFormat("0");
        writer.write("Test set, predicted values:\n");
        for (int i = 0; i < test_act_ho.length; i++) {
            writer.write(rounded.format(test_act_ho[i]));

            if (i < test_act_ho.length - 1) {
                writer.write(",");
            } else {
                writer.write("\n");
            }
        }

        // find closest to 0.5 activation value
        double minDifference = 100;
        int minIndex = -1;
        for (int i = 0; i < test_act_ho.length; i++) {
            if (Math.abs(test_act_ho[i] - 0.5) < minDifference) {
                minDifference = test_act_ho[i];
                minIndex = i;
            }
        }
        writer.write("Test image index closest to 0.5: " + minIndex + "\n");

        // print feature vector of test image closest to 0.5
        DecimalFormat twoPlaces = new DecimalFormat("0.00");
        writer.write("Feature vector of test image closest to 0.5: \n");

        for (int i = 0; i < test[minIndex].length; i++) {
            writer.write(twoPlaces.format(test[minIndex][i]));

            if (i < test[minIndex].length - 1) {
                writer.write(",");
            } else {
                writer.write("\n");
            }
        }
    }

    private static void printWeights(FileWriter writer, double[][] wih, double[] who) throws IOException {
        DecimalFormat df = new DecimalFormat("0.0000");

        writer.write("First layer, 784x392 weights, 1x392 bias:\n");
        for (int i = 0; i < 785; i++) {
            for (int j = 0; j < 392; j++) {
                writer.write(df.format(wih[j][i]));
                if (j < wih.length - 1) {
                    writer.write(",");
                } else {
                    writer.write("\n");
                }
            }
        }

        writer.write("Second layer, 392 weights, 1 bias:\n");
        for (int i = 0; i < who.length; i++) {
            writer.write(df.format(who[i]));
            if (i < who.length - 1) {
                writer.write(",");
            } else {
                writer.write("\n");
            }
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

