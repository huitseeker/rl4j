package org.deeplearning4j.rl4j.network.dqn;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Value;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.rl4j.util.Constants;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/13/16.
 */
@Value
public class DQNFactoryStdConv implements DQNFactory {


    Configuration conf;

    public DQN buildDQN(int shapeInputs[], int numOutputs) {

        if (shapeInputs.length == 1)
            throw new AssertionError("Impossible to apply convolutional layer on a shape == 1");


        NeuralNetConfiguration.ListBuilder confB = new NeuralNetConfiguration.Builder().seed(Constants.NEURAL_NET_SEED)
                        .iterations(1).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .learningRate(conf.getLearningRate())
                        .regularization(conf.getL2() > 0)
                        .l2(conf.getL2())
                        //.updater(Updater.NESTEROVS).momentum(0.9)
                        //.updater(Updater.RMSPROP).rmsDecay(conf.getRmsDecay())
                        .updater(conf.getUpdater() != null ? conf.getUpdater() : new Adam())
                        .weightInit(WeightInit.XAVIER).regularization(true).l2(conf.getL2()).list()
                        .layer(0, new ConvolutionLayer.Builder(8, 8).nIn(shapeInputs[0]).nOut(16).stride(4, 4)
                                        .activation(Activation.RELU).build());


        confB.layer(1, new ConvolutionLayer.Builder(4, 4).nOut(32).stride(2, 2).activation(Activation.RELU).build());

        confB.layer(2, new DenseLayer.Builder().nOut(256).activation(Activation.RELU).build());

        confB.layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nOut(numOutputs)
                        .build());

        confB.setInputType(InputType.convolutional(shapeInputs[1], shapeInputs[2], shapeInputs[0]));
        MultiLayerConfiguration mlnconf = confB.pretrain(false).backprop(true).build();
        MultiLayerNetwork model = new MultiLayerNetwork(mlnconf);
        model.init();
        if (conf.getListeners() != null) {
            model.setListeners(conf.getListeners());
        } else {
            model.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER));
        }

        return new DQN(model);
    }


    @AllArgsConstructor
    @Builder
    @Value
    public static class Configuration {

        double learningRate;
        double l2;
        IUpdater updater;
        IterationListener[] listeners;
    }

}
