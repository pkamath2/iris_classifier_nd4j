package org.pk.iris;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 * Author: purnimakamath
 */
public class DataLoader {

    public static INDArray loadData(){

        try(FileReader fileReader = new FileReader("/Users/purnimakamath/appdir/Github/iris_classifier_nd4j/src/main/resources/iris.csv");
            BufferedReader bufferedReader = new BufferedReader(fileReader)){

            float[][] data = new float[150][5];
            String line = bufferedReader.readLine();
            int row = 0, col = 0;
            String num;
            while(line != null){
                String[] data_arr = line.split(",");
                for (String str:data_arr) {
                    num = str.trim();
                    if(col == 4){
                        switch (str){
                            case "I. virginica": num = "3"; break;
                            case "I. versicolor" : num = "2"; break;
                            case "I. setosa" : num = "1";
                        }
                    }
                    data[row][col] = Float.parseFloat(num);
                    col++;
                }

                line = bufferedReader.readLine();
                row++;
                col = 0;
            }
            return Nd4j.create(data);

        }catch (IOException ioe){
            System.err.println(ioe.getMessage());
        }
        return null;
    }


    public static DataSet[] train_test_split(DataSet original_dataset){
        // 65-35 split
        original_dataset.shuffle(0);
        SplitTestAndTrain stt = original_dataset.splitTestAndTrain(0.65);

        DataSet[] dataSets = new DataSet[2];
        dataSets[0] = stt.getTrain();
        dataSets[1] = stt.getTest();

        return dataSets;
    }


    public static INDArray convertNumLabelsToSoftmaxLabels(INDArray labels){//Labels is 1Xm array

        INDArray label_arr = Nd4j.create(labels.shape()[1],3);
        INDArray label_lookup= Nd4j.eye(3);

        for(int i=0;i<labels.shape()[1];i++){
            INDArray label = label_lookup.getRow(labels.getInt(i)-1);
            label_arr.putRow(i, label);

        }
        return label_arr;
    }
}
