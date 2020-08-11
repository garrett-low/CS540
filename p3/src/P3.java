import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class P3 {

    private static char[] alphabet = " abcdefghijklmnopqrstuvwxyz".toCharArray(); // space + alphabet
    private static int lengthScript = 0;                                          // this gets updated later
    private static final int SENTENCE_LENGTH = 1000;                              // characters

    private static Map<String, Integer> count_unigrams = new HashMap<String, Integer>(); // count of unigrams
    private static Map<String, Integer> count_bigrams = new HashMap<String, Integer>();  // count of bigrams
    private static Map<String, Integer> count_trigrams = new HashMap<String, Integer>(); // count of trigrams

    private static Map<String, Double> transitionProbability1 = new HashMap<String, Double>(); // P(x)
    private static Map<String, Double> transitionProbability2 = new HashMap<String, Double>(); // P(y|x)
    private static Map<String, Double> transitionProbability3 = new HashMap<String, Double>(); // P(z|xy)

    private static final String outputPath = "./output.txt";
    private static final DecimalFormat fourPlaces = new DecimalFormat("0.000");

    public static void main(String[] args) throws IOException {
        createFile(outputPath);

        try (FileWriter writer = new FileWriter(outputPath)) {
            // Step 1 - read a script of the movie from txt file into String object
            String script = new String(Files.readAllBytes(Paths.get("./silence_of_the_lambs_script.txt")));
            System.out.println("Finished reading the input file with script. ");

            // Step 2 - processing the script
            // make everything lowercase, remove other chars excepts letter or space with a single space
            script = script.toLowerCase().replaceAll("[^a-z ]", " ").replaceAll(" +", " ");
            System.out.println("Finished processing the script. ");

            // store the value of total chars in the processed script, will need this later
            lengthScript = script.length();

            // Step 3 - this step takes some time to run
            countNGrams(script);
            System.out.println("Counted all unigrams, bigrams and trigrams for the script.");

            // Step 4 - estimate transition probabilities (MLE) without Laplace smoothing
            estimateTransitionProbabilities(1, count_unigrams, transitionProbability1);
            estimateTransitionProbabilities(2, count_bigrams, transitionProbability2);
            estimateTransitionProbabilities(3, count_trigrams, transitionProbability3);
            // Q2 - print unigram probabilities
            writer.write("Q2 - unigram probabilities:\n");
            System.out.println("Q2 - unigram probabilities:");
            printList(transitionProbability1, fourPlaces, writer);
            // Q3 - print bigram probabilities
            writer.write("Q3 - bigram probabilities:\n");
            System.out.println("Q3 - bigram probabilities:");
            ArrayList<Double> bigramProbability = mapToArray(transitionProbability2);
            ArrayList<Double> roundedBigramProbability = roundDouble(bigramProbability, 10000.0);
            ArrayList<Double> fixRowSumBigram = fixRowSum(roundedBigramProbability);
            printArray(fixRowSumBigram, fourPlaces, writer);

            writer.write("Q3 - bigram probabilities:\n");
            System.out.println("Q3 - bigram probabilities:");
            printList(transitionProbability2, fourPlaces, writer);
            writer.write(transitionProbability2.toString());
            System.out.println(transitionProbability2);

            // Step 5 - generate sentences using trigram model, in some cases bigram model is used as well
            System.out.println("================== Generating sentences for each letter ==================");
            generateSentences();
            System.out.println("==========================================================================");


            // Step 6 - compute likelihood prob and posterior probabilities for the script given by Young
            String scriptYoung = new String(Files.readAllBytes(Paths.get("./script_grlow.txt")));

            // process this script as well
            scriptYoung = scriptYoung.toLowerCase().replaceAll("[^a-z ]", " ").replaceAll(" +", " ");

            Map<String, Integer> count_unigramsScript2 = new HashMap<String, Integer>();
            int count = 0;
            for (int i = 0; i < alphabet.length; i++) {
                count = scriptYoung.length() - scriptYoung.replace(String.valueOf(alphabet[i]), "").length();
                count_unigramsScript2.put(String.valueOf(alphabet[i]), count);
            }

            Map<String, Double> transitionProbabilityScript2 = new HashMap<String, Double>(); // P(x)
            double probability;
            for (String key : count_unigramsScript2.keySet()) {
                probability = (count_unigramsScript2.get(key)) / (double) (scriptYoung.length());
                transitionProbabilityScript2.put(key, probability);
            }
            System.out.println("Likelihood probabilities for the script of Young:");
            System.out.println(transitionProbabilityScript2);

            // computing posterior probabilities for the script of Young
            Map<String, Double> posteriorProb = new HashMap<String, Double>();
            double post_prob;
            for (String key : count_unigramsScript2.keySet()) {
                post_prob = transitionProbabilityScript2.get(key) / (transitionProbabilityScript2.get(key) +
                        transitionProbability1.get(key));
                posteriorProb.put(key, post_prob);
            }
            System.out.println("Posterior probabilities for the script of Young:");
            System.out.println(posteriorProb);

            // TODO: similarly compute posterior probabilities for our script
            // TODO: need to implement Laplace smoothing, otherwise current code might not work correctly (division by 0)
            // TODO: need to implement Naive Bayes prediction
        }
    }

    private static ArrayList<Double> fixRowSum(ArrayList<Double> probabilityArray) {
        Double sumExceptLast = 0.0;
        for (int i = 0; i < probabilityArray.size(); i++) {
            if ((i + 1) % 27 == 0) {
                probabilityArray.set(i, 1 - sumExceptLast);
                sumExceptLast = 0.0;
            } else if (i < probabilityArray.size() - 1) {
                sumExceptLast += probabilityArray.get(i);
            }
        }

        return probabilityArray;
    }

    private static ArrayList<Double> roundDouble(ArrayList<Double> probabilityArray, Double places) {
        for (int i = 0; i < probabilityArray.size(); i++) {
            probabilityArray.set(i, (Math.floor(probabilityArray.get(i) * places) / places));
        }
        return probabilityArray;
    }

    private static ArrayList<Double> mapToArray(Map<String, Double> transitionProbability) {
        ArrayList<Double> probabilityArray = new ArrayList<>();
        transitionProbability.forEach((k, v) -> probabilityArray.add(v));
        return probabilityArray;
    }

    private static void printList(Map<String, Double> transitionProbability, DecimalFormat df, FileWriter writer) throws IOException {
        ArrayList<Double> probabilityArray = new ArrayList<>();
        transitionProbability.forEach((k, v) -> probabilityArray.add(v));
        for (int i = 0; i < probabilityArray.size(); i++) {
            writer.write(df.format(probabilityArray.get(i)));
            System.out.print(df.format(probabilityArray.get(i)));

            if ((i + 1) % 27 == 0) {
                writer.write("\n");
                System.out.println();
            } else if (i < probabilityArray.size() - 1) {
                writer.write(",");
                System.out.print(",");
            }
        }
    }

    // I forgot how to use generics
    private static void printArray(ArrayList<Double> probabilityArray, DecimalFormat df, FileWriter writer) throws IOException {
        for (int i = 0; i < probabilityArray.size(); i++) {
            writer.write(df.format(probabilityArray.get(i)));
            System.out.print(df.format(probabilityArray.get(i)));

            if ((i + 1) % 27 == 0) {
                writer.write("\n");
                System.out.println();
            } else if (i < probabilityArray.size() - 1) {
                writer.write(",");
                System.out.print(",");
            }
        }
    }

    public static void countNGrams(String script) {

        // count # of occurrences of each char in an alphabet + space character
        int count = 0;
        for (int i = 0; i < alphabet.length; i++) {
            count = script.length() - script.replace(String.valueOf(alphabet[i]), "").length();
            count_unigrams.put(String.valueOf(alphabet[i]), count);
        }

        // count # of occurrences for each bigram
        for (int i = 0; i < alphabet.length; i++) {
            for (int j = 0; j < alphabet.length; j++) {
                count = (script.length() - script.replace(String.valueOf(alphabet[i]) +
                        String.valueOf(alphabet[j]), "").length()) / 2;
                count_bigrams.put(String.valueOf(alphabet[i]) + String.valueOf(alphabet[j]), count);
            }
        }

        // count # of occurrences for each trigram
        for (int i = 0; i < alphabet.length; i++) {
            for (int j = 0; j < alphabet.length; j++) {
                for (int k = 0; k < alphabet.length; k++) {
                    count = (script.length() - script.replace(String.valueOf(alphabet[i]) +
                            String.valueOf(alphabet[j]) + String.valueOf(alphabet[k]), "").length()) / 3;
                    count_trigrams.put(String.valueOf(alphabet[i]) + String.valueOf(alphabet[j]) +
                            String.valueOf(alphabet[k]), count);
                }
            }
        }
    }

    /*
     * TODO: need to change this method to use Laplace smoothing
     */
    public static void estimateTransitionProbabilities(int n, Map<String, Integer> ngram_count, Map<String, Double> trans_prob) {

        double probability;
        for (String key : ngram_count.keySet()) {
            if (n == 1) {
                // compute P(x)
                probability = (ngram_count.get(key)) / (double) (lengthScript);
            } else if (n == 2) {
                // compute P (y | x)
                probability = (ngram_count.get(key)) / (double) (count_unigrams.get(String.valueOf(key.charAt(0))));
            } else {
                // compute P(z | xy)
                probability = (ngram_count.get(key)) / (double) (count_bigrams.get(key.substring(0, 2)));
            }
            trans_prob.put(key, probability);
        }
    }

    public static void generateSentences() {

        // go through each letter, ignore first char which is space
        for (int i = 1; i < alphabet.length; i++) {

            StringBuilder sb = new StringBuilder();
            sb.append(String.valueOf(alphabet[i])); // append the first letter

            // keep building a sentence until length is 1000 chars
            while (sb.length() < SENTENCE_LENGTH) {

                double[] cdf;

                if (sb.length() == 1 || transitionProbability2.get(sb.toString().substring(sb.length() - 2, sb.length())) == 0.0) {
                    // for 2nd letter compute CDF using bigram or when can't use trigram model in general
                    cdf = computeCDF(String.valueOf(sb.toString().charAt(0)), transitionProbability2);
                } else {
                    // for the rest of the letters compute CDF using trigram model
                    cdf = computeCDF(sb.toString().substring(sb.length() - 2, sb.length()), transitionProbability3);
                }

                // generate random uniform variable between 0 and max value of the cdf
                Random r = new Random();
                double randomValue = cdf[cdf.length - 1] * r.nextDouble();

                char letter = findNextLetter(randomValue, cdf); // found next letter/space

                sb.append(letter); // append next letter to the sentence
            }
            // print out the generated sentence
            System.out.println(sb.toString());
        }
    }

    public static double[] computeCDF(String c, Map<String, Double> trans_prob) {

        double[] cdf = new double[27];
        double sum = 0;
        for (int i = 0; i < 27; i++) {
            sum = sum + trans_prob.get(c + String.valueOf(alphabet[i]));
            cdf[i] = sum;
        }
        return cdf;
    }

    public static char findNextLetter(double u, double[] cdf) {

        for (int i = 0; i < alphabet.length; i++) {
            if (u <= cdf[i]) {
                return alphabet[i];
            }
        }
        // if greater than sum of all probabilities then return last char, shouldn't happen though
        return 'z';
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
