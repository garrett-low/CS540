import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.lang.Math;



// Todo: you need to change the activation function from relu (current) version to logistic, remember, not only the activation function, but the weight update part as well.

public class NN_student {

    // Todo: change hyper-parameters below, like MAX_EPOCHS, learning_rate, etc.

    private static final String COMMA_DELIMITER = ",";
    private static final String PATH_TO_TRAIN = "./mnist_train.csv";
    private static final String PATH_TO_TEST = "./mnist_test.csv";
    private static final String NEW_TEST = "./test.txt";
    private static final int MAX_EPOCHS = 1;
    static Double learning_rate = 0.1;

    static Double[][] wih = new Double[392][785];
    static Double[] who = new Double[393];

    static String first_digit = "7";
    static String second_digit = "4";
    static Random rng = new Random();


    public static double[][] parseRecords(String file_path) throws FileNotFoundException, IOException {
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
            for (int i= 0; i < k; i ++){
                System.arraycopy(records[i], 0, res[i], 0, 786);
            }
            return res;
        }

    }


    public static double[][] NewTest(String file_path) throws FileNotFoundException, IOException {
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
            for (int i= 0; i < k; i ++){
                System.arraycopy(records[i], 0, res[i], 0, 785);
            }
            return res;
        }

    }


    public static double relu(double x){
        if (x > 0) return x;
        else return 0.0;
    }

    public static double diff_relu(double x){
        if (x > 0 ) return 1.0;
        else return 0.0;
    }


    public static void main(String[] args) throws IOException {
        double[][] train = parseRecords(PATH_TO_TRAIN);
        double[][] test = parseRecords(PATH_TO_TEST);

        double[][] new_test = NewTest(NEW_TEST);


        int num_train = train.length;
        int num_test = test.length;

        for(int i = 0; i < wih.length; i ++){
            for (int j = 0; j < wih[0].length; j++){
                wih[i][j] = 2 * rng.nextDouble() - 1;
            }
        }
        for(int i = 0; i < who.length; i ++){
            who[i] = 2 * rng.nextDouble() - 1;
        }


        for(int epoch = 1; epoch <= MAX_EPOCHS; epoch ++ ){
            double[] out_o = new double[num_train];
            double[][] out_h = new double[num_train][393];
            for(int i = 0; i < num_train; ++ i)
                out_h[i][392] = 1.0;

            for(int ind = 0; ind < num_train; ++ ind){
                double[] row = train[ind];
                double label = row[0];


                //calc out_h[ind, :-1]
                for(int i = 0; i < 392; ++ i) {
                    double s = 0.0;
                    for (int j = 0; j < 785; ++j) {
                        s += wih[i][j] * row[j+1];
                    }
                    out_h[ind][i] = relu(s);
                }

                //calc out_o[ind]
                double s = 0.0;
                for(int i = 0; i < 393; ++ i){
                    s += out_h[ind][i] * who[i];
                }
                out_o[ind] = 1.0 / (1.0 + Math.exp(-s));

                //calc delta
                double[] delta = new double[393];
                for(int i = 0; i < 393; ++i){
                    delta[i] = diff_relu(out_h[ind][i]) * who[i] * (label - out_o[ind]);
                }

                //update wih
                for(int i = 0; i < 392; ++i){
                    for(int j = 0; j < 785; ++ j){
                        wih[i][j] += learning_rate * delta[i] * row[j+1];
                    }
                }

                //update who
                for(int i = 0; i < 393; ++ i){
                    who[i] += learning_rate * (label - out_o[ind]) * out_h[ind][i];
                }
            }


            //calc error
            double error = 0;
            for(int ind = 0; ind < num_train; ind ++){
                double[] row = train[ind];
                error += -row[0] * Math.log(out_o[ind]) - (1-row[0]) * Math.log(1- out_o[ind]);
            }

            //correct
            double correct = 0.0;
            for(int ind = 0; ind < num_train; ind ++){
                if ((train[ind][0] == 1.0 && out_o[ind] >=0.5) || (train[ind][0] ==0.0 && out_o[ind] < 0.5) )
                    correct += 1.0;
            }
            double acc = correct / num_train;

            System.out.println("Epoch: " + epoch + ", error: " + error + ", acc: " + acc);

        }


        // Todo: your new_test. Hint: use above 'new_test'




    }



}

