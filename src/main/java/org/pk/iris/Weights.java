package org.pk.iris;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Author: purnimakamath
 */
public class Weights {

    private INDArray W1;
    private INDArray W2;
    private INDArray W3;
    private INDArray W4;
    private INDArray b1;
    private INDArray b2;
    private INDArray b3;
    private INDArray b4;

    public Weights() {
    }

    public Weights(INDArray w1, INDArray w2, INDArray w3, INDArray w4, INDArray b1, INDArray b2, INDArray b3, INDArray b4) {
        W1 = w1;
        W2 = w2;
        W3 = w3;
        W4 = w4;
        this.b1 = b1;
        this.b2 = b2;
        this.b3 = b3;
        this.b4 = b4;
    }

    public INDArray getW1() {
        return W1;
    }

    public void setW1(INDArray w1) {
        W1 = w1;
    }

    public INDArray getW2() {
        return W2;
    }

    public void setW2(INDArray w2) {
        W2 = w2;
    }

    public INDArray getW3() {
        return W3;
    }

    public void setW3(INDArray w3) {
        W3 = w3;
    }

    public INDArray getW4() {
        return W4;
    }

    public void setW4(INDArray w4) {
        W4 = w4;
    }

    public INDArray getB1() {
        return b1;
    }

    public void setB1(INDArray b1) {
        this.b1 = b1;
    }

    public INDArray getB2() {
        return b2;
    }

    public void setB2(INDArray b2) {
        this.b2 = b2;
    }

    public INDArray getB3() {
        return b3;
    }

    public void setB3(INDArray b3) {
        this.b3 = b3;
    }

    public INDArray getB4() {
        return b4;
    }

    public void setB4(INDArray b4) {
        this.b4 = b4;
    }
}
