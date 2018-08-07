package org.pk.iris;

import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

import static org.pk.iris.DataLoader.*;

/**
 * Author: purnimakamath
 */
public class IrisClassifier {

    private static Logger log = LoggerFactory.getLogger(IrisClassifier.class);

    public static void main(String[] args) {

        //1. Load Data from file
        INDArray orig_data = loadData();
        DataSet orig_dataset = new DataSet(orig_data.getColumns(0,1,2,3), orig_data.getColumn(4));
        orig_dataset.shuffle();

        //2. Split data into Train and Test sets
        DataSet[] dataSets = train_test_split(orig_dataset);

        INDArray X_orig = orig_dataset.getFeatures();
        INDArray y_orig = orig_dataset.getLabels();
        y_orig = convertNumLabelsToSoftmaxLabels(y_orig.transpose());

        DataSet train_data = dataSets[0];
        INDArray X_train = train_data.getFeatures();
        INDArray y_train = train_data.getLabels();
        y_train = convertNumLabelsToSoftmaxLabels(y_train.transpose());

        DataSet test_data = dataSets[0];
        INDArray X_test = test_data.getFeatures();
        INDArray y_test = test_data.getLabels();
        y_test = convertNumLabelsToSoftmaxLabels(y_test.transpose());

        // 3. Initialize weights
        Weights weights = initializeWeights();

        // 4. Fit & Predict
        log.debug("Fitting.......");
        log.debug("Starting Shapes X_train & y_train shapes => " + Arrays.toString(X_train.shape())+", "+Arrays.toString(y_train.shape()));
        Map<String, INDArray> params = fitOrPredict(X_train, y_train, weights, X_train.shape()[0],0.1,10);

        log.debug("Predicting.......");
        Map output = fitOrPredict(X_test, y_test, new Weights(params.get("W1"),params.get("W2"),params.get("W3"),params.get("W4"),
                params.get("b1"),params.get("b2"),params.get("b3"),params.get("b4")),X_test.shape()[0],0.1,1);
        log.debug("Predicted output - " + output.get("output"));

        // 5. Print metrics
        Evaluation eval = new Evaluation(3);
        eval.eval(y_test, (INDArray)output.get("output"));
        log.info(eval.stats());
    }


    private static HashMap<String, INDArray> fitOrPredict(INDArray X, INDArray y, Weights weights, int num_samples, double learning_rate, int epoch){
        HashMap<String, INDArray> params = new HashMap<>();

        X = X.transpose(); //Transpose to fit the math Z = WX + b
        y = y.transpose();

        INDArray W1 = weights.getW1();
        INDArray W2 = weights.getW2();
        INDArray W3 = weights.getW3();
        INDArray W4 = weights.getW4();

        INDArray b1 = weights.getB1();
        INDArray b2 = weights.getB2();
        INDArray b3 = weights.getB3();
        INDArray b4 = weights.getB4();

        List<Double> costList = new ArrayList<>();
        INDArray A4 = null;
        while(epoch > 0) {
            INDArray Z1 = W1.mmul(X).addColumnVector(b1);
            INDArray A1 = Transforms.tanh(Z1);

            INDArray Z2 = W2.mmul(A1).addColumnVector(b2);
            INDArray A2 = Transforms.tanh(Z2);

            INDArray Z3 = W3.mmul(A2).addColumnVector(b3);
            INDArray A3 = Transforms.tanh(Z3);

            INDArray Z4 = W4.mmul(A3).addColumnVector(b4);
            A4 = Transforms.softmax(Z4.transpose()); //Softmax works out of rows.
            A4 = A4.transpose();

            //TODO: Compute cost equation needs to be corrected
            double cost = (-1) * Nd4j.sum(Transforms.log(A4, Math.E).mul(y), 0).getDouble(0);
            costList.add(cost);

            INDArray dZ4 = A4.sub(y);
            INDArray dW4 = dZ4.mmul(A3.transpose()).mul(1/num_samples);
            INDArray db4 = Nd4j.mean(dZ4, 1);

            INDArray dZ3 = W4.mmul(dZ4).mul((Transforms.pow(A3, 2).mul(-1)).add(1));
            INDArray dW3 = dZ3.mmul(A2.transpose()).mul(1 / num_samples);
            INDArray db3 = Nd4j.mean(dZ3, 1);

            INDArray dZ2 = W3.mmul(dZ3).mul((Transforms.pow(A2, 2).mul(-1)).add(1));
            INDArray dW2 = dZ2.mmul(A1.transpose()).mul(1 / num_samples);
            INDArray db2 = Nd4j.mean(dZ2, 1);

            INDArray dZ1 = W2.mmul(dZ2).mul((Transforms.pow(A1, 2).mul(-1)).add(1));
            INDArray dW1 = dZ1.mmul(X.transpose()).mul(1 / num_samples);
            INDArray db1 = Nd4j.mean(dZ1, 1);

            W1 = W1.sub(dW1.mul(learning_rate));
            b1 = b1.sub(db1.mul(learning_rate));

            W2 = W2.sub(dW2.mul(learning_rate));
            b2 = b2.sub(db2.mul(learning_rate));

            W3 = W3.sub(dW3.mul(learning_rate));
            b3 = b3.sub(db3.mul(learning_rate));

            W4 = W4.sub(dW4.mul(learning_rate));
            b4 = b4.sub(db4.mul(learning_rate));

            epoch--;
        }

        params.put("W1", W1);
        params.put("W2", W2);
        params.put("W3", W3);
        params.put("W4", W4);
        params.put("b1",b1);
        params.put("b2",b2);
        params.put("b3",b3);
        params.put("b4",b4);
        params.put("output",A4);
        return params;
    }


    private static Weights initializeWeights(){
        Weights weights = new Weights();
        Nd4j.getRandom().setSeed(100);

        INDArray W1 = Nd4j.rand(new int[]{3, 4}).muli(FastMath.sqrt(2.0D/(4+3)));
        INDArray b1 = Nd4j.zeros(3, 1);

        INDArray W2 = Nd4j.rand(new int[]{3, 3}).muli(FastMath.sqrt(2.0D/(3+3)));
        INDArray b2 = Nd4j.zeros(3, 1);

        INDArray W3 = Nd4j.rand(new int[]{3, 3}).muli(FastMath.sqrt(2.0D/(3+3)));
        INDArray b3 = Nd4j.zeros(3, 1);

        INDArray W4 = Nd4j.rand(new int[]{3, 3}).muli(FastMath.sqrt(2.0D/(3+3)));
        INDArray b4 = Nd4j.zeros(3, 1);

        weights.setB1(b1);
        weights.setB2(b2);
        weights.setB3(b3);
        weights.setB4(b4);

        weights.setW1(W1);
        weights.setW2(W2);
        weights.setW3(W3);
        weights.setW4(W4);

        return weights;
    }


}
